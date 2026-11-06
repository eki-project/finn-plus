# Zero-cost proxies (`finn.nas.zc`)

Training-free ("zero-cost") proxy scores for ranking NAS candidates without training them,
vendored from NASLib and adapted to Brevitas quantization-aware-training networks.

```python
from finn.nas.zc import DEFAULT_MEASURES, available_measures, compute_proxies

res = compute_proxies(brevitas_module, images, labels, num_classes=10)   # one minibatch
res.scores    # {"nwot": 612.5, "synflow": 8.5e4, ..., "params": 33940.0, "flops": 2.8e6}
res.errors    # {"jacov": "LinAlgError: ..."}   only for measures that failed (score = NaN)
res.seconds   # wall-clock time per measure
res.to_dict() # plain dictionaries, JSON-serializable
```

`compute_proxies(model, inputs, targets, measures=None, *, num_classes, loss_fn=None,
device=None, seed=0)` evaluates `measures` (default: `DEFAULT_MEASURES`, i.e. the proxies that
work on Brevitas networks plus `params` and `flops`) and returns a `ProxyResult`. The loss
defaults to cross entropy, the device to the device of the model. `available_measures()` lists
everything that can be requested, including the measures that are not in the default list.

- The model that is passed in is never modified: every measure runs on a deep copy that is put
  into training mode (batch norm and the Brevitas activation quantizers of an untrained network
  then take their statistics/scales from the minibatch; `synflow` bypasses batch norm), and on a
  private copy of the batch. The training flag, parameters, buffers and `.grad` of the caller's
  model are untouched.
- Every measure runs isolated. If it raises or returns a non-finite value, its score is NaN and
  `errors` holds a one-line reason; the other measures are unaffected.
- `torch.manual_seed(seed)` is called before every measure; the CPU and CUDA random states are
  restored afterwards and cuDNN runs deterministically inside the call, so the caller's random
  stream is not disturbed and two calls give identical scores (verified on CPU and CUDA). Only
  `zen` draws random numbers. Scores on CPU and CUDA agree to ~1e-6 relative (zen excepted, the
  random inputs differ per device).
- `l2_norm` reads the weights of the model in place (on the model's device, whatever `device`
  says); it does not modify them.
- One `gc.collect()` runs at the end of `compute_proxies` (the hooks some measures install keep
  the copies alive through reference cycles); on CUDA additionally after every measure.
- `params` is the number of trainable parameters (for Brevitas this includes learned quantizer
  scales), `flops` the number of multiply-accumulates of all `Conv1d/2d/3d` and `Linear` layers
  for one sample (bias, batch norm, pooling and activations are not counted).

## Origin and license

Vendored from NASLib, https://github.com/automl/NASLib, branch `zerocost`
(the NAS-Bench-Suite-Zero code), commit `acf6fa43770d7caa7cea0712440b2e9013bf7b07`
(2024-11-11), directory `naslib/predictors/pruners/`. NASLib is licensed under the Apache
License, Version 2.0 (Copyright 2019-2021 AutoML Freiburg and contributors). Most of the
vendored files originate from Samsung's zero-cost-nas (https://github.com/SamsungLabs/zero-cost-nas,
Copyright 2021 Samsung Electronics Co., Ltd., Apache-2.0) and carry Samsung's header;
`epe_nas.py` is MIT licensed (Copyright (c) 2021 VascoLopes) and `zen.py` carries only the
copyright line of Alibaba Group Holding Limited (2010-2021), it is redistributed here as received
from NASLib. All original copyright and license headers are kept in the files; see `NOTICE` in
this directory for the attribution notices and the text of the Apache License. NASLib itself is
not a dependency, nothing else of it is used, and no new third-party dependency was added
(torch, numpy and brevitas only).

| vendored file (`naslib/predictors/pruners/`) | here (`src/finn/nas/zc/`) | license header |
|---|---|---|
| `measures/__init__.py` | `measures/__init__.py` | Samsung, Apache-2.0 |
| `p_utils.py` | `measures/p_utils.py` | Samsung, Apache-2.0 |
| `predictive.py` | `measures/predictive.py` (reduced, see below) | Samsung, Apache-2.0 |
| `measures/epe_nas.py` | `measures/epe_nas.py` | VascoLopes, MIT |
| `measures/fisher.py` | `measures/fisher.py` | Samsung, Apache-2.0 |
| `measures/grad_norm.py` | `measures/grad_norm.py` | Samsung, Apache-2.0 |
| `measures/grasp.py` | `measures/grasp.py` | Samsung, Apache-2.0 |
| `measures/jacov.py` | `measures/jacov.py` | Samsung, Apache-2.0 |
| `measures/l2_norm.py` | `measures/l2_norm.py` | Samsung, Apache-2.0 |
| `measures/nwot.py` | `measures/nwot.py` | Samsung, Apache-2.0 |
| `measures/plain.py` | `measures/plain.py` | Samsung, Apache-2.0 |
| `measures/snip.py` | `measures/snip.py` | Samsung, Apache-2.0 |
| `measures/synflow.py` | `measures/synflow.py` | Samsung, Apache-2.0 |
| `measures/zen.py` | `measures/zen.py` | Alibaba copyright line |

Not vendored: `measures/model_stats.py` (needs tensorwatch; `params`/`flops` are computed in
`proxies.py` instead), the data loader helpers `get_some_data*` and the `find_measures*` driver
of `predictive.py` (data loading, CUDA out-of-memory retry with batch splitting, and it moves the
caller's model to the CPU and into training mode), `pruners/__init__.py` and NASLib's `ZeroCost`
predictor class. `measures/brevitas_compat.py` and `proxies.py` are FINN+ code, not vendored.

## Local modifications

All vendored files: imports changed to the new package layout (`..p_utils` -> `.p_utils`),
`from __future__ import annotations`, type hints and docstrings added (the project requires
them for every function), formatted with black/isort (line length 100), one-letter variable
`l` renamed to `layer`, a file-level `# ruff: noqa` for the rules the NASLib signatures would
violate (unused measure arguments `ARG001`, upper-case locals `N803/N806`, `import ... as F`
`N812`, the original module docstrings of `jacov.py`/`nwot.py` `D212/D400/D404/D415`, the
`input` builtin in `zen.py` `A001`). Each modified file has a "Modified for FINN+" note below
its original header. Beyond that:

- `measures/__init__.py`: the copy is made by `copynet(net_orig, bn)` directly instead of a
  `get_prunable_copy` method that NASLib attaches to the caller's model; the copy is put into
  training mode; the forced `gc.collect()` after every measure only runs on CUDA (it costs about
  0.2 s in a FINN process, `compute_proxies` collects once at the end instead); `load_all`
  imports the measure modules through `importlib`.
- `measures/predictive.py`: reduced to `no_op`, `copynet` and `sum_arr` (the score reduction of
  `find_measures`, lifted to module level).
- `measures/p_utils.py`: reduced to `get_layer_metric_array`, `reshape_elements`,
  `count_parameters`; `type(x) == list` -> `isinstance`, `zip(..., strict=False)`.
- `jacov.py`, `epe_nas.py`: the `try/except` that printed an exception and returned NaN is
  removed, failures reach `compute_proxies`, which records them; the bare `except:` in
  `eval_score_perclass` is `except Exception:`.
- `nwot.py`: hooks the modules returned by `brevitas_compat.activation_modules` instead of
  every module whose type name contains "ReLU"; skips an activation that is fed directly by the
  network input; unwraps `QuantTensor` inputs. See the Brevitas section.
- `snip.py`: Brevitas layers keep their forward, the mask is applied to the latent weight on its
  way into the weight quantizer (`snip_mask_quant_weight`). Float layers unchanged.
- `fisher.py`: Brevitas layers keep their forward and only get the recording identity appended
  (`fisher_forward_quant`, `QuantTensor` outputs supported); the torch warning about the
  deprecated non-full backward hook is silenced locally (the hook only uses `grad_output`,
  which that hook reports correctly). Float layers unchanged.
- `synflow.py`: the quantizers of the copy are switched off (`brevitas_compat.
  disable_quantization`) before the measure runs, i.e. the float skeleton of a Brevitas network
  is scored. See the Brevitas section.
- `zen.py`: networks without `forward_before_global_avg_pool` are evaluated on the input of
  their last `nn.Linear` (`forward_before_classifier`), i.e. after global pooling/flattening,
  and the perturbation is summed over all non-batch dimensions (the original assumed a 4-D
  feature map).

## Brevitas networks

The measures were written for float networks. Brevitas `QuantConv2d`/`QuantLinear` are
subclasses of `nn.Conv2d`/`nn.Linear`, so the layer selection of all measures finds them, and
their forward quantizes input, weight, bias and output with straight-through gradients. What
was checked per measure (float CNN and Brevitas CNNs with 1/2/4/8-bit weights and activations,
a residual network, a bipolar CNV-like network, `return_quant_tensor=True` layers; CIFAR-sized
batch of 64; all scores finite unless stated; "non-degenerate" means the score differs between
two clearly different architectures and is not constant or zero; "time" is the CPU time of one
measure for a 3-conv-layer Brevitas CNN, 16/32/64 channels, 32x32 inputs, batch 64):

| measure | runs on float | runs on Brevitas | finite, non-degenerate on Brevitas | adaptation | time | default |
|---|---|---|---|---|---|---|
| `grad_norm` | yes | yes | yes | none; gradient of the latent weight through the quantizers | 0.08 s | yes |
| `snip` | yes | yes (unadapted: bypassed the quantizers) | yes | mask in front of the weight quantizer; equals sum of abs(weight x grad) (verified) | 0.08 s | yes |
| `grasp` | yes | yes | yes, but init-seed noise is as large as the differences between architectures | none; double backward works through the Brevitas STE functions | 0.19 s | yes |
| `fisher` | yes | yes (unadapted: bypassed the quantizers) | yes | records the output of the unchanged quantized forward | 0.09 s | yes |
| `jacov` | yes | yes | yes (small spread, -65 +- 2, as in float) | none | 0.08 s | yes |
| `plain` | yes | yes | yes, but init-seed noise is as large as the differences between architectures | none | 0.10 s | yes |
| `synflow` | yes | no (unadapted: `KeyError`, the Brevitas activation scaler hides its `value` from `state_dict()` until it has seen a batch, so the key set changes between linearize and nonlinearize; with that fixed, bipolar activations zero the score of every layer in front of them) | yes | scored on the float skeleton (quantizers disabled) | 0.03 s | yes |
| `synflow_bn` | yes (not with `BatchNorm1d`) | as `synflow` | finite, but batch norm in training mode on a single all-ones sample makes it meaningless (per-layer values spread over 7 orders of magnitude, also for float networks) | as `synflow` | 0.03 s | no |
| `epe_nas` | yes | yes | yes | none | 0.07 s | yes |
| `zen` | yes (needs `forward_before_global_avg_pool`; otherwise the fallback) | no (unadapted: no `forward_before_global_avg_pool`) | finite, but at <= 2-bit activations the finite difference of a piecewise-constant function is noise: std over seeds 0.8-1.3 vs. differences between architectures of ~1 | fallback features, generic reduction | 0.10 s | no |
| `l2_norm` | yes | yes | yes | none; norm of the latent float weights | 0.001 s | yes |
| `nwot` | yes | yes (unadapted: counted every `QuantReLU` twice, -inf without ReLU) | yes | activation selection, see below | 0.06 s | yes |
| `params` | yes | yes | yes | FINN+ code | 0 s | yes |
| `flops` | yes | yes | yes | FINN+ code | 0.02 s | yes |

- Activation-like modules (`nwot`): the float ReLU variants NASLib selects (class name contains
  "ReLU") and all Brevitas quantized activation layers (`QuantReLU`, `QuantIdentity`,
  `QuantHardTanh`, ...), each once: the `nn.ReLU` a `QuantReLU` wraps is skipped, it would count
  the same activation twice (an offset of `batch x log 2` per activation). `QuantIdentity` is
  included because in low-precision FINN networks the quantizer is the nonlinearity (a bipolar
  `QuantIdentity` is a sign activation); the binary code is `input > 0` in all cases. An
  activation fed directly by the network input (the usual input quantizer) is skipped, its code
  depends only on the data. With this, a 4-bit network and its float twin (same topology) score
  within 0.5 of each other (batch of 64); the double counting shifted the score by 44.
- `synflow`: takes the absolute value of every `state_dict` entry (also quantizer scales and
  counters, harmless on a copy, but a scale stored in the log domain would flip) and feeds a
  single all-ones sample. It assumes a positively homogeneous network; quantizers are not
  (rounding, clipping), and a bipolar activation blocks the straight-through gradient of all
  layers in front of it, so with the quantized forward only the classifier of a CNV-like
  network contributes. For 2- to 8-bit networks the quantized forward agrees with the float
  skeleton to a few percent (per-layer conservation holds), so the float skeleton is used for all
  Brevitas networks. The score is therefore blind to the bit widths.
- `snip`/`fisher`: the NASLib versions replace the layer forward by `F.conv2d`/`F.linear`, which
  drops all quantizers. For Brevitas layers the quantized forward is kept; the SNIP mask
  multiplies the latent weight in front of the weight quantizer (removing a latent weight is what
  pruning a QAT network means), the Fisher hook records the quantized layer output.
- Surprise worth knowing: a forward pass of a layer with Brevitas' clamped binary weight
  quantizer (`SignedBinaryWeightPerTensorConst`) clamps the latent weight parameter in place,
  also in evaluation mode. This is why every measure must run on a copy; the tests cover it.
- At initialization most proxies hardly react to the bit width (1/2/4/8-bit variants of one
  topology score within the init-seed noise); they rank topologies, not precisions.

None of this says anything yet about how well these proxies predict the accuracy of a network
after quantization-aware training. That has to be measured on the search space at hand.

## Tests

`tests/nas/test_zc.py` (`finn test --variant custom --args "$PWD/tests/nas/test_zc.py"`): finite
and architecture-dependent scores on two Brevitas networks for every default measure, all
measures on a float network, model untouched (state, training flag, grads, input, global RNG
state) for float, Brevitas and bipolar networks, determinism, failure isolation, `params` and
`flops` by hand, `synflow` on a bipolar network, the activation selection of `nwot`.
