# Grammar-based search space for FINN-aware NAS

`finn.nas.space` is an einspace-style search space ("Searching for Neural Architectures from
Fundamental Operations", Ericsson et al., NeurIPS 2024): architectures are sampled from a
recursive probabilistic context-free grammar whose terminals are the layers FINN can in
principle map. The space is deliberately *not* restricted to what FINN accepts; learning
FINN's constraints from the oracle's feedback is the job of the search algorithm. Sampling,
validation and variation are pure Python (no torch); only `build.py` imports torch/Brevitas.

```python
import random
from finn.nas.space import SearchSpace, SpaceConfig, describe, genotype_hash
from finn.nas.space.build import to_module          # torch + brevitas
from finn.nas import FinnOracle, OracleConfig

space = SearchSpace(SpaceConfig())                  # CIFAR-10, <= 2 M parameters
rng = random.Random(0)
parent = space.sample(rng)
child = space.mutate(parent, rng)
sibling_a, sibling_b = space.crossover(parent, child, rng)
print(describe(child))
result = FinnOracle(OracleConfig(fpga_part="xc7z020clg400-1")).evaluate(
    to_module(child), input_shape=(1, 3, 32, 32)
)
```

## Grammar

```
body       -> sequential                                  (the root is always a chain)
module     -> sequential | branching | terminal           module_weights
sequential -> module module [module [module]]             sequential_lengths
branching  -> (module, module [, module]; add | concat)   branch_counts, agg_weights
terminal   -> conv | linear | quant_act | relu_float | batchnorm
            | maxpool | avgpool | flatten | identity      terminal_weights
```

Hyperparameters of the terminals are drawn uniformly from the choice lists of the config
(`SpaceConfig.choices(kind)`). Defaults:

| terminal | hyperparameters | applicable |
|---|---|---|
| `conv` | kernel {1,3,5}, stride {1,2}, padding {same, valid}, out_channels {8..128} or depthwise (p = 0.25), weight_bits {1,2,3,4,8}; no bias; in_channels follow from the input | image mode, kernel fits |
| `linear` | out_features {16..512}, weight_bits | vector mode |
| `quant_act` | bits {1,2,3,4,8}: 1 = bipolar sign, else unsigned quantized ReLU | always |
| `relu_float` | - (plain `nn.ReLU`; FINN maps it almost nowhere, the search should learn that) | always |
| `batchnorm` | - (2d / 1d by mode) | always |
| `maxpool` | kernel {2,3}, stride {1,2} | image mode, kernel fits |
| `avgpool` | kernel {2,3}, stride = kernel | image mode, kernel fits |
| `flatten` | - | image mode |
| `identity` | - (as a branch: skip connection) | always |

Aggregations: `add` needs equal branch shapes, `concat` needs equal mode and equal H, W
(channels resp. features are concatenated). Around the body sit a fixed input quantizer
(`input_bits`) and a classification head (`head.kind` in {gap, flatten}, `head.weight_bits`):
global average pooling if the body ends in image mode and the kind is `gap`, otherwise a
flatten, then a quantized linear layer to `num_classes` returning float logits. Both head
genes and `input_bits` are part of the genotype.

The only rules in the sampler and the variation operators are the ones that make a network
well-formed in PyTorch (kernel/pool fits the current spatial size, equal shapes for `add`,
linear only in vector mode, flatten only in image mode, spatial size >= 1) and the budgets of
the config (`max_depth`, `max_nodes`, `max_params`). Nothing filters by divisibility,
bit-width combinations, layer order or anything else FINN cares about.

## Genotype

Plain JSON data; no derived information (shapes, input channels, parameter counts) is stored.

```json
{"version": 1, "input_shape": [3, 32, 32], "num_classes": 10, "input_bits": 8,
 "body": {"kind": "sequential", "children": [
     {"kind": "conv", "kernel": 3, "stride": 1, "padding": "same", "depthwise": false,
      "out_channels": 16, "weight_bits": 4},
     {"kind": "batchnorm"}, {"kind": "quant_act", "bits": 4},
     {"kind": "branching", "agg": "add", "children": [{"kind": "identity"}, {"...": "..."}]}]},
 "head": {"kind": "flatten", "weight_bits": 4}}
```

- Node kinds: `sequential` (`children`), `branching` (`agg`, `children`) and the terminals
  with exactly their hyperparameters as fields (a depthwise conv has no `out_channels`).
  Exact field sets keep the canonical form of a network unique.
- `canonical_json(g)` (sorted keys, no whitespace) and `genotype_hash(g)` (sha256 hex) are
  stable across processes; `json.loads(json.dumps(g)) == g`.
- Nodes are addressed by *paths*, tuples of child indices from the body root `()`.
- `infer_shapes(g)` / `validate(g)` compute in/out shape, parameter count and subtree size of
  every node analytically and raise `InvalidGenotypeError` (with the node path and reason)
  for ill-formed genotypes. `validate(g, config)` additionally checks membership in the space
  (task, budgets, every value in its choice list). Shapes are `(C, H, W)` in image mode and
  `(F,)` in vector mode.
- `count_params(g)` counts conv/linear weights (no biases), the scale and shift of batch
  normalization and the head's weights; quantizers have constant scales and contribute
  nothing. It equals `sum(p.numel() for p in to_module(g).parameters())`.
- `describe(g)` renders one indented line per node with its output shape, for logs.
- Head kind after a body that ends in vector mode is a neutral gene (two genotypes that only
  differ in it build the same network but have different hashes).

## Sampling and variation (`SearchSpace`)

`sample(rng)` recurses through the grammar like einspace: the current shape and the remaining
node/parameter budget are carried down, only applicable productions are offered (a terminal is
offered when at least one hyperparameter setting is applicable; dense/depthwise conv variants
are enumerated and weighted so that each choice list stays uniform), a production that dead-ends
is removed and another one is drawn (backtracking), and the whole body is restarted if the
network with its head exceeds a budget. Later branches of a `branching` node are rejection-
sampled (`branch_match_attempts`) until they can be aggregated with the first branch; an
identity branch is a legal result. `SamplingError` is raised only after `max_sample_attempts`
restarts. All randomness comes from the `random.Random` argument.

`mutate(g, rng)` either resamples a random subtree for the same input shape (einspace's
mutation; the replacement may use the budget the old subtree had) or changes one
hyperparameter of a random terminal or of the wrapper (`input_bits`, head genes), with
probability `subtree_mutation_prob`. `crossover(a, b, rng)` swaps a random subtree of `a` with
a random subtree of `b` and returns both children. Both re-validate the whole genotype against
the config and retry up to `max_variation_attempts` times; mutation returns a copy of the
parent if nothing else validates, crossover falls back to mutating both parents. Inputs are
never modified.

## Module construction (`build.py`)

`to_module(g)` returns a `GenotypeNet` mapping an `(N, C, H, W)` float batch to
`(N, num_classes)` logits. Quantizers follow the BNN-PYNQ reference models (`CommonWeightQuant`
/ `CommonActQuant`), which FINN is known to export and streamline:

- Weights: constant scale 1 (weights in [-1, 1]), narrow-range integers, binary (bipolar) at
  1 bit. Weights are initialized uniformly in [-1, 1] as BNN-PYNQ does, because the default
  fan-in scaled initialization would round to zero at low bit widths.
- Input quantizer: signed fixed point in [-1, 1) with scale `2**-(bits-1)` (Q1.7 for 8 bits);
  inputs are expected normalized to that range. 1 bit: sign.
- Activations: 1 bit is a bipolar sign; otherwise an unsigned quantized ReLU saturating at
  `2**bits - 1`, i.e. **the activations are the integers 0 .. 2**bits-1 and the scale is
  exactly 1**. This generalizes BNN-PYNQ's 2-bit activations (-1, 0, 1). It matters for
  branching: FINN's default streamlining cannot move a scale factor across a fork or a join,
  so with any other range no multi-bit activation could feed a residual block. A preceding
  batch normalization adapts the range during training.
- Constant scales also mean that the exported graph structure, and thus FINN's verdict,
  depends only on the genotype and not on learned scale factors (a search would otherwise be
  told "feasible" at initialization and "infeasible" after training).
- Average pooling: hardware cannot divide by the window size, so FINN maps only Brevitas'
  truncating `TruncAvgPool2d` (sum, drop low-order bits). The builder tracks statically which
  tensors are known to be integer (output of the input quantizer or a multi-bit activation,
  through identity, max pooling and truncating average pooling) and builds `avgpool`
  terminals and the `gap` head as `TruncAvgPool` for those (`IntFormat` attaches the format
  locally, so no `QuantTensor` objects flow through the network). Elsewhere they are plain
  `nn.AvgPool2d` / `nn.AdaptiveAvgPool2d`, which FINN rejects. Pooling a 1x1 map is built as
  a flatten.
- `branching` is an `nn.Module` summing or concatenating its branches; `sequential` is
  `nn.Sequential`; the module tree mirrors the genotype tree (useful for hooks).

## Seeds (`seeds.py`)

Hand-written genotypes inside the default space, all four FINN-compatible with the oracle:
`cnv_like` (scaled-down BNN-PYNQ CNV, w2a2), `mlp_like` (flatten + three FC blocks, w2a2),
`residual` (stem, identity skip + two conv blocks, add, max pool, flatten head, w4a4) and
`depthwise_separable` (stem, two strided depthwise + pointwise blocks, gap head, w4a4).
`tests/nas/test_space_oracle.py` documents genotype-level neighbours that FINN rejects
(a skip forking the input quantizer's output; global average pooling of a float tensor).

## Defaults and budgets

`SpaceConfig()` describes CIFAR-10 classifiers: input `(3, 32, 32)`, 10 classes,
`max_params = 2e6` (CNV has 1.5 M), `max_nodes = 40` tree nodes (CNV has 28 operations),
`max_depth = 6`. The module weights (sequential 0.3, branching 0.15, terminal 0.55) give a
little more than one child per node on average, so the recursion is bounded by the budgets
rather than dying out early; the terminal weights (conv 3, linear 3, quant_act 3, batchnorm 2,
relu_float 1, maxpool 1, avgpool 1, flatten 0.5, identity 1) roughly follow how often the
operations occur in hand-designed quantized networks and encode no FINN knowledge. With these
defaults the median sample has ~15 nodes, about 60 % of samples contain a branching node and
about a quarter end in vector mode. Sampling costs a few milliseconds per genotype.

## Files

- `config.py` – `SpaceConfig` (task, budgets, weights, choice lists, retry limits).
- `genotype.py` – constructors, canonical JSON, hashing, path utilities, `InvalidGenotypeError`.
- `shapes.py` – `apply_terminal`, `infer_shapes`, `validate`, `count_params`.
- `sampler.py` – `Sampler` (recursive sampling), `SamplingError`.
- `variation.py` – `mutate`, `crossover`.
- `search_space.py` – `SearchSpace` facade.
- `describe.py` – `describe`.
- `seeds.py` – seed genotypes.
- `build.py` – `to_module`, quantizers, `TruncAvgPool` (the only module importing torch).

Tests: `tests/nas/test_space.py` (torch-free), `tests/nas/test_space_build.py` (torch),
`tests/nas/test_space_oracle.py` (oracle; the random-sample smoke test is marked `slow`).
