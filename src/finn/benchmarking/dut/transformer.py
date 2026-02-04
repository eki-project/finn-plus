"""Transformer benchmarking module adapted from Christoph's attention-dummy repository."""

# PyTorch base package: Math and Tensor Stuff
import json
import numpy as np
import random
import torch
from brevitas.export import export_qonnx

# Brevitas: Quantized versions of PyTorch layers
from brevitas.nn import (
    QuantEltwiseAdd,
    QuantIdentity,
    QuantLinear,
    QuantMultiheadAttention,
    QuantReLU,
)

# Brevitas wrapper around PyTorch tensors adding quantization information
from brevitas.quant_tensor import QuantTensor
from qonnx.core.modelwrapper import ModelWrapper

# Progressbar
from tqdm import trange

# FINN dataflow builder
import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.bench_base import bench

# Custom build steps required to streamline and convert the attention operator
from finn.builder.custom_step_library.transformer import (
    prepare_graph,
    set_fifo_depths,
    set_target_parallelization,
    step_apply_folding_config,
    step_convert_attention_to_hw,
    step_convert_depth_wise_to_hw,
    step_convert_elementwise_binary_to_hw,
    step_convert_lookup_to_hw,
    step_convert_split_concat_to_hw,
    step_replicate_streams,
    step_streamline,
)


# ADAPTED FROM utils.py
def seed(s):
    """Seeds all relevant random number generators to the same seed for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ADAPTED FROM model.py
def weight_quantizer(bits, _signed=True):
    """Derives a weight quantizer, leaving bit-width and signedness configurable."""
    # Brevitas quantizer base classes
    from brevitas.inject.enum import RestrictValueType
    from brevitas.quant.base import MaxStatsScaling, NarrowIntQuant
    from brevitas.quant.solver import WeightQuantSolver

    # Derive a Quantizer from the brevitas bases
    class Quantizer(NarrowIntQuant, MaxStatsScaling, WeightQuantSolver):
        """Weight quantizer with configurable bit-width and signedness."""

        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


def bias_quantizer(bits, _signed=True):
    """Derives a bias quantizer, leaving bit-width and signedness configurable."""
    # Brevitas quantizer base classes
    from brevitas.quant import IntBias

    # Derive a Quantizer from the brevitas bases
    class Quantizer(IntBias):
        """Bias quantizer with configurable bit-width and signedness."""

        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Do not require the bit-width to be adjusted to fit the accumulator to
        # which the bias is added
        requires_input_bit_width = False

    # Return the derived quantizer configuration
    return Quantizer


def act_quantizer(bits, _signed=True):
    """Derives an activation quantizer, leaving bit-width and signedness configurable."""
    # Brevitas quantizer base classes
    from brevitas.inject.enum import RestrictValueType
    from brevitas.quant.base import IntQuant, ParamFromRuntimePercentileScaling
    from brevitas.quant.solver import ActQuantSolver

    # Derive a Quantizer from the brevitas bases
    class Quantizer(IntQuant, ParamFromRuntimePercentileScaling, ActQuantSolver):
        """Activation quantizer with configurable bit-width and signedness."""

        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


def get_norm(key, normalized_shape):
    """Gets the normalization layer from configuration key."""

    class Transpose(torch.nn.Module):
        """Transposes Sequence and Embedding dimensions."""

        def forward(self, x):  # noqa: May be static
            """Forward pass transposing the feature map."""
            # Transpose the last two dimensions of batch x seq x emb layout
            return torch.transpose(x, dim0=-1, dim1=-2)

    # Dictionary mapping keys to supported normalization layer implementations
    norms = {
        # PyTorch default layer normalization. Needs to know the shape of the
        # feature map to be normalized
        "layer-norm": torch.nn.LayerNorm(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            normalized_shape=normalized_shape,
            elementwise_affine=False,
        ),
        # PyTorch default 1-dimensional batch normalization. Needs to transpose
        # embedding and sequence dimension to normalized over the embedding
        # dimension, which is expected to be second.
        "batch-norm": torch.nn.Sequential(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            Transpose(),
            torch.nn.LazyBatchNorm1d(affine=False),
            Transpose(),
        ),
        # No normalization by a PyTorch built-in identity layer. Should not
        # appear in the graph.
        "none": torch.nn.Identity(),
    }

    # Select the normalization layer by key
    return norms[key]


def get_mask(key, length):
    """Gets the attention mask from configuration key and shape."""
    # Dictionary mapping keys to supported normalization layer implementations
    masks = {
        # No attention mask
        "none": None,
        # Generate the upper triangular mask for causal attention
        "causal": torch.nn.Transformer.generate_square_subsequent_mask(length),
        # Square matrix with entries randomly set to -inf or 0.0 with 50%
        # probability each
        "random": torch.where(  # noqa: Confused by types?
            torch.rand(length, length) > 0.5, -torch.inf, 0.0
        ),
    }
    # Select the mask type by key
    return masks[key]


class TransformerBlock(torch.nn.Module):
    """Single-layer scaled dot-product attention block with MLP and normalization."""

    def __init__(self, num_heads, emb_dim, mlp_dim, seq_len, bias, norm, mask, bits):
        """Initializes the model and registers the module parameters."""
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Input quantizer to the scaled dot-product attention operations, shared
        # by queries, keys and values inputs. It is important to have this
        # quantizer separate and not preceding the fork node of the residual
        # branches to avoid consecutive quantizers in the skip branch.
        # Note: For some reason it seems not to be possible to use the
        #   in_proj_input_quant of the attention operator
        self.sdp_input_quant = QuantIdentity(
            # Quantize at the output
            act_quant=act_quantizer(bits, _signed=True),
            # Pass quantization information on to the next layer.
            return_quant_tensor=True,
        )
        # Quantized scaled dot-product attention operator
        self.sdp = QuantMultiheadAttention(
            # Size of the embedding dimension (input and output)
            embed_dim=emb_dim,
            # Number of attention heads
            num_heads=num_heads,
            # Enable a bias added to the input and output projections
            bias=bias,
            # Layout of the inputs:
            #   Batch x Sequence x Embedding (batch-first, True)
            #   Sequence x Batch x Embedding (batch-second, False)
            batch_first=True,
            # If query, key and value input are the same, packed input
            # projections use a single, large linear projection to produce
            # the actual query, key and value inputs. Otherwise, use
            # separate linear projections on each individual input.
            packed_in_proj=False,
            # Brevitas has this as an unsigned quantizer by default, but
            # finn can only handle signed quantizer
            attn_output_weights_quant=act_quantizer(bits, _signed=True),
            # Insert an additional quantizer in front ot the softmax. In our
            # finn custom-op, this will be matched to the quantizer
            # following the query and key matmul.
            # Note: Disable to prevent the quantizer from tripping over -inf
            # from the attention mask
            softmax_input_quant=None,
            # Quantize the input projections weights as configured
            in_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the input projections as configured
            in_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # No quantization in front of the input projections as this is
            # either done by a standalone quantizer preceding the whole block
            in_proj_input_quant=None,
            # Quantize the output projections weights as configured
            out_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the output projections as configured
            out_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # Quantize the input to the output projection as configured
            out_proj_input_quant=act_quantizer(bits, _signed=True),
            # Quantizer the key after projections as configured
            k_transposed_quant=act_quantizer(bits, _signed=True),
            # Quantize the queries after projections as configured
            q_scaled_quant=act_quantizer(bits, _signed=True),
            # Quantize the values after projection as configured
            v_quant=act_quantizer(bits, _signed=True),
            # No output quantization for now, as stacking multiple layers
            # results in multiple multi-thresholds in succession
            out_proj_output_quant=None,
            # Return the quantization parameters so the next layer can
            # quantize the bias
            return_quant_tensor=True,
        )
        # Residual branch addition skipping over the attention layer
        self.residual_sdp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            return_quant_tensor=True,
        )
        # Normalization following the attention layer
        self.norm_sdp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the MLP residual
            # branch. See input quantizer in front of the first MLP layer.
        )

        # Quantized MLP following the scaled dot-product attention
        self.mlp = torch.nn.Sequential(
            # Quantize the inputs to the MLP block. Placed here to not have this
            # at the input of the residual branch.
            QuantIdentity(
                # Quantize at the output
                act_quant=act_quantizer(bits, _signed=True),
                # Pass quantization information on to the next layer.
                return_quant_tensor=True,
            ),
            # First mlp layer projecting to the mlp dimension
            QuantLinear(
                # Inputs have the size of the attention embedding dimension
                emb_dim,
                # Project to the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as this is directly preceded by a
                # standalone quantizer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized ReLU activation taking care of quantization
                output_quant=None,
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True,
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            QuantReLU(
                # Note: ReLU must be quantized to unsigned representation
                act_quant=act_quantizer(bits, _signed=False),
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True,
            ),
            # Second mlp layer projecting back to the embedding dimension
            QuantLinear(
                # Inputs have the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Project back to the size of the attention embedding dimension
                emb_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as the inputs are already quantized by the
                # preceding ReLU layer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized element-wise addition taking care of quantization
                output_quant=None,
                # Pass quantization information on to the next layer.
                return_quant_tensor=True,
            ),
        )
        # Residual branch addition skipping over the MLP layer
        self.residual_mlp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            # Note: Not for the last layer to allow this to be combined with
            # standard pytorch calls like .detach() or .numpy(), which are
            # not directly available on QuantTensor.
            return_quant_tensor=True,
        )
        # Normalization following the attention layer
        self.norm_mlp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the SDP residual
            # branch
        )
        # Generate the attention mask according to configuration
        self.mask = get_mask(mask, seq_len)

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        """Forward pass adding sinusoidal positional encoding to input tensor."""
        """Forward pass through the transformer block."""
        # Move the mask to the same device as the input, just in case...
        mask = self.mask.to(x.device) if self.mask is not None else None
        # Quantize the input to the attention block
        q = self.sdp_input_quant(x)
        # Scaled dot-product attention with residual branch and normalization
        x = self.norm_sdp(self.residual_sdp(x, self.sdp(q, q, q, attn_mask=mask)[0]))
        # MLP layer with residual branch and normalization
        return self.norm_mlp(self.residual_mlp(x, self.mlp(x)))


class QuantSinusoidalPositionalEncoding(torch.nn.Module):
    """Quantized sinusoidal positional encoding layer."""

    def __init__(self, input_quant, output_quant, return_quant_tensor):
        """Initializes the model and registers the module parameters."""
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x):
        """Forward pass adding sinusoidal positional encoding to input tensor."""
        # Get the size of the inputs to dynamically generate encodings of the
        # same size
        _, seq, emb = x.shape
        # Start by enumerating all steps of the sequence
        i = torch.as_tensor([[n] for n in range(seq)])
        # Scale factor adjusting the frequency/wavelength of the sinusoid
        # depending on the embedding dimension index
        f = torch.as_tensor([1e4 ** -(i / emb) for i in range(0, emb, 2)])
        # Prepare empty positional encoding tensor of the same size as the input
        pos = torch.empty(seq, emb)
        # Fill the positional encoding with alternating sine and cosine waves
        pos[:, 0::2] = torch.sin(f * i)
        pos[:, 1::2] = torch.cos(f * i)
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=x.dtype)
        # Add the quantized encoding to the quantized input
        return self.add(x, pos)


class QuantLearnedPositionalEncoding(torch.nn.Module):
    """Quantized learned positional encoding layer."""

    def __init__(self, seq_len, emb_dim, input_quant, output_quant, return_quant_tensor):
        """Initializes the model and registers the module parameters."""
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor,
        )
        # Register a parameter tensor representing the not quantized positional
        # encoding
        self.pos = torch.nn.Parameter(torch.empty(seq_len, emb_dim))
        # Reset/Initialize the parameter tensor
        self.reset_parameters()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        """Resets/Initializes positional encoding parameters from normal distribution."""
        # Initialize the positional encoding from a normal distribution with
        # zero mean and unit standard deviation
        torch.nn.init.normal_(self.pos, mean=0, std=1)

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        """Forward pass adding learned positional encoding to input tensor."""
        # Add the quantized encoding to the quantized input
        return self.add(x, self.pos)


class LazyQuantLearnedPositionalEncoding(
    torch.nn.modules.lazy.LazyModuleMixin, QuantLearnedPositionalEncoding  # noqa
):
    """
    Lazy version of the learned encoding not requiring input dimensions at initialization,
    inferring these at the first forward pass.
    """

    # Once initialized, this will become a QuantLearnedPositionalEncoding as
    # defined above
    cls_to_become = QuantLearnedPositionalEncoding
    # Parameter tensor of the QuantLearnedPositionalEncoding is uninitialized
    pos: torch.nn.UninitializedParameter

    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        """Initializes lazy learned positional encoding with uninitialized parameters."""
        # Initialize the quantizer parts of QuantLearnedPositionalEncoding,
        # leaving the dimensions empty
        super().__init__(0, 0, input_quant, output_quant, return_quant_tensor)
        # Register an uninitialized parameter tensor for the positional encoding
        self.pos = torch.nn.UninitializedParameter()

    # Resets/Initializes the positional encoding parameter tensor
    def reset_parameters(self):
        """Resets parameters if already initialized, otherwise delegates to parent."""
        # If this has already been initialized, delegate to the actual
        # implementation
        if not self.has_uninitialized_params():
            super().reset_parameters()

    # Initializes/Materializes the uninitialized parameter tensor given some
    # sample input tensor to infer the dimensions
    def initialize_parameters(self, x):
        """Materializes uninitialized parameter tensor by inferring dimensions from input."""
        # Only materialize the parameter tensor if it is not yet initialized
        if self.has_uninitialized_params():
            # Do not accumulate gradient information from initialization
            with torch.no_grad():
                # Get the size of the inputs to generate encodings of the same
                # size
                _, seq, emb = x.shape
                # Materialize the positional encoding parameter tensor
                self.pos.materialize((seq, emb))
                # Properly initialize the parameters by resetting the values
                self.reset_parameters()


class QuantBinaryPositionalEncoding(torch.nn.Module):
    """Quantized binary positional encoding layer."""

    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        """Initializes binary positional encoding with quantized addition."""
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor,
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        """Forward pass adding binary positional encoding to input tensor."""
        # Get the size of the inputs to dynamically generate encodings of the
        # same size
        _, seq, emb = x.shape
        # Binary positional encoding fills the embedding dimension with the bit
        # pattern corresponding to the position in the sequence
        pos = torch.as_tensor([[(n & (1 << bit)) >> bit for bit in range(emb)] for n in range(seq)])
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=x.dtype)
        # Add the quantized encoding tp the quantized input
        #   Note: Convert encoding to bipolar representation
        return self.add(x, 2 * pos - 1)


def get_positional_encoding(key, input_quant, output_quant, return_quant_tensor):
    """Gets the positional encoding layer from configuration key, quantizers and shape."""
    # Dictionary mapping keys to supported normalization layer implementations
    masks = {
        # No positional encoding
        "none": QuantIdentity(act_quant=input_quant, return_quant_tensor=return_quant_tensor),
        # Fixed, sinusoidal positional encoding according to Vaswani et al. with
        # added quantizers
        "sinusoidal": QuantSinusoidalPositionalEncoding(
            input_quant, output_quant, return_quant_tensor
        ),
        # Fixed, binary positional encoding with quantizers
        "binary": QuantBinaryPositionalEncoding(input_quant, output_quant, return_quant_tensor),
        # Learned positional encoding with quantizers
        "learned": LazyQuantLearnedPositionalEncoding(
            input_quant, output_quant, return_quant_tensor
        ),
    }
    # Select the positional encoding type by key
    return masks[key]


def unpack_from_quant(tensor: torch.Tensor | QuantTensor):
    """Unpacks the standard PyTorch tensor from a brevitas QuantTensor."""
    # If this is a QuantTensor we can extract the wrapped tensor
    if isinstance(tensor, QuantTensor):
        # The underlying tensor is wrapped as the value attribute
        return tensor.value
    # Assume this is already a plain PyTorch tensor
    return tensor


class DummyTransformer(torch.nn.Module):
    """Dummy transformer encoder model."""

    def __init__(
        self,
        # Number of layers of attention blocks
        num_layers,
        # Number of attention heads per block
        num_heads,
        # Size of embedding dimension going into/out of the attention block
        emb_dim,
        # Size of MLP dimension in each attention block
        mlp_dim,
        # Length of the input sequence, i.e., context size
        seq_len,
        # Enables bias term added to Linear layers
        bias,
        # Quantization bit-width: For now all layers are quantized to the
        # same bit-width
        bits,
        # Type of normalization layer to use in the transformer blocks
        #   Options are: layer-norm, batch-norm and none
        norm="none",
        # Type of attention mask to use
        #   Options are: none, causal or const
        mask="none",
        # Type of positional encoding to use at the input
        #   Options are: none, sinusoidal, binary, learned
        positional_encoding="none",
    ):
        """Initializes the model and registers the module parameters."""
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Positional encoding layer at the input
        self.pos = get_positional_encoding(
            # Select the implementation by configuration key
            key=positional_encoding,
            # Quantize the inputs to the positional encoding to the same
            # bit-width as the input
            input_quant=act_quantizer(bits, _signed=True),
            # Quantize the sum of input and positional encoding to the same
            # bit-width as the input
            output_quant=None,
            # Pass quantization information on to the next layer
            return_quant_tensor=True,
        )

        # Sequence of num_layers transformer encoder blocks
        self.encoder = torch.nn.Sequential(
            *[
                TransformerBlock(num_heads, emb_dim, mlp_dim, seq_len, bias, norm, mask, bits)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """Model forward pass."""
        # Add positional encoding to the input and feed through the encoder
        # stack
        # Note: Get the wrapped value out of the QuantTensor to have only a
        # single output from the model.
        return unpack_from_quant(self.encoder(self.pos(x)))


# ADAPTED FROM export.py


def is_norm_layer(module):
    """Check whether a layer is a normalization layer of some supported type."""
    # Set of normalization layer (bases) which maybe need to be patched
    norm_layers = {
        # All BatchNorm and InstanceNorm variants derive from this baseclass
        torch.nn.modules.batchnorm._NormBase,  # noqa: Access to _NormBase
        # LayerNorm has a unique implementation
        torch.nn.LayerNorm,
    }
    # Check the module against all supported norm layer types
    return any(isinstance(module, norm) for norm in norm_layers)


def patch_non_affine_norms(model: torch.nn.Module):  # noqa: Shadows model
    """Fixes export issues of normalization layers with disabled affine parameters."""
    # Somehow the export to ONNX trips when it encounters the weight and bias tensor
    # to be 'None'.
    # Iterate all modules in the model container
    for name, module in model.named_modules():
        # If the module is a normalization layer it might require patching the
        # affine parameters
        if is_norm_layer(module):
            # Check whether affine scale parameters are missing
            if hasattr(module, "weight") and module.weight is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_var"):
                    # Patch the affine bias by all 1 tensor of the same shape,
                    # type and device as the running variance
                    module.weight = torch.nn.Parameter(torch.ones_like(module.running_var))
            # Check whether affine bias parameters are missing
            if hasattr(module, "bias") and module.bias is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_mean"):
                    # Patch the affine bias by all 0 tensor of the same shape,
                    # type and device as the running mean
                    module.bias = torch.nn.Parameter(torch.zeros_like(module.running_var))
    # Return the patched model container
    return model


template_folding_yaml = """
# Per operator type default configurations
defaults:
    # Scaled dot-product attention head implemented via HLS
    ScaledDotProductAttention_hls:
        # Type of memory to be used for internal buffer storage
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Type of memory to be used for threshold storage
        #   Options: auto, block, distributed
        ram_style_thresholds: block
        # Type of memory to be used fo the attention mask (if present)
        #   Options: auto, block, distributed
        ram_style_mask: block
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        mac_resource: lut
    # Addition of two inputs (constants or streamed) implemented via HLS
    ElementwiseAdd_hls:
        # Type of memory to be used for internal buffer storage and/or constant
        # parameter tensors
        #   Options: auto, block, distributed, ultra
        ram_style: distributed
    # Matrix vector activation unit implemented via HLS
    MVAU_hls:
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        resType: dsp
        # Memory mode for weight storage
        #   Options: internal_embedded, internal_decoupled, external
        mem_mode: internal_decoupled
        # Type of memory to be used for weight storage if "internal_decoupled"
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Type of memory to be used for threshold storage
        #   Options: auto, block, distributed
        ram_style_thresholds: block
        # Makes weights writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Matrix vector activation unit implemented via RTL
    MVAU_rtl:
        # Resource type to be used for implementing multiplications/MACs
        #   Options: auto, lut or dsp
        # Note: RTL MVAU currently does not support LUT-based implementation
        resType: dsp
        # Memory mode for weight storage
        #   Options: internal_embedded, internal_decoupled, external
        mem_mode: internal_decoupled
        # Type of memory to be used for weight storage if "internal_decoupled"
        #   Options: auto, block, distributed, ultra
        ram_style: block
        # Makes weights writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Multi-thresholds implemented via HLS (applies to standalone thresholds)
    Thresholding_hls:
        # Memory mode for threshold storage
        #   Options: internal_embedded, internal_decoupled
        mem_mode: internal_decoupled
        # Type of memory to be used for threshold storage if "internal_decoupled"
        #   Options: distributed, block
        ram_style: distributed
        # Makes thresholds writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # Multi-thresholds implemented via RTL (applies to standalone thresholds)
    Thresholding_rtl:
        # Decides to use BRAM, URAM or LUTs for threshold memory, depending on the
        # depth of the thresholds
        # Note: This combination forces "distributed" LUT implementation
        depth_trigger_uram: 2147483647  # "infinity"
        depth_trigger_bram: 2147483647  # "infinity"
    #    # Note: This combination forces "block" RAM implementation
    #    depth_trigger_uram: 0
    #    depth_trigger_bram: 1
    #    # Note: This combination forces "ultra" RAM implementation
    #    depth_trigger_uram: 1
    #    depth_trigger_bram: 0
    #    # Note: This combination is equivalent to "auto"
    #    depth_trigger_uram: 0
    #    depth_trigger_bram: 0
        # Makes thresholds writeable through AXI-lite interface at runtime
        runtime_writeable_weights: 0
    # FIFO implemented via RTL (there is no HLS FIFO implementation in FINN)
    StreamingFIFO_rtl:
        # RTL vs. IPI implementation of FIFOs
        #   Options: rtl, vivado
        impl_style: rtl
        # Resource type for FIFOs when impl_style is vivado
        #   Options: auto, block, distributed, ultra
        ram_style: distributed
    # Individual, named node-specific configurations here
    # ...
"""


class bench_transformer(bench):
    """Benchmark class for quantized transformer models."""

    def _step_export_onnx(self, output_onnx_path):
        """Generates a dummy transformer block, not used for actual models (RadioML, GPT, etc.)."""
        # Generates a dummy transformer block,
        # not used for actual models (RadioML, GPT, etc.)

        # Load the parameters file
        # params = dvc.api.params_show("params.yaml")
        # Seed all RNGs
        seed(self._params["seed"])
        # Make PyTorch behave deterministically if possible
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        # Create a model instance from the configuration parameters
        # model = DummyTransformer(**params["model"])
        model = DummyTransformer(
            num_layers=self._params["model_num_layers"],
            num_heads=self._params["model_num_heads"],
            emb_dim=self._params["model_emb_dim"],
            mlp_dim=self._params["model_mlp_dim"],
            seq_len=self._params["model_seq_len"],
            bias=self._params["model_bias"],
            bits=self._params["model_bits"],
            norm=self._params["model_norm"],
            mask=self._params["model_mask"],
            positional_encoding=self._params["model_positional_encoding"],
        )

        # Get the configured sequence length and embedding dimension to generate
        # test inputs
        seq, dim = self._params["model_seq_len"], self._params["model_emb_dim"]
        # No gradient accumulation for calibration passes required
        with torch.no_grad():
            # Check whether GPU training is available and select the appropriate
            # device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Move the model to the training device
            model = model.to(device)
            # Multiple passes of calibration might be necessary for larger/deep
            # models
            for _ in trange(0, self._params["calibration_passes"], desc="calibrating"):
                # Pass random data through the model to "calibrate" dummy quantizer.
                # Large batch to have more calibration samples. Otherwise, there is
                # too much deviation between this calibration and the verification
                # samples.
                model(torch.rand(128, seq, dim, device=device))
            # Move the model back to the CPU
            model = model.cpu()
        # Prevent export issue for missing affine normalization parameters
        model = patch_non_affine_norms(model)
        # Switch model to evaluation mode to have it fixed for export
        model = model.eval()
        # Sample random input tensor in batch-first layout
        x = torch.rand(1, seq, dim)
        # Compute attention output
        o = model(x)
        # Save the input and output data for verification purposes later
        np.save("inp.npy", x.detach().numpy())
        np.save("out.npy", o.detach().numpy())
        self._build_inputs["input_npy_path"] = "inp.npy"
        self._build_inputs["output_npy_path"] = "out.npy"
        # Export the model graph to QONNX
        # export_qonnx(model, (x,), "attention.onnx", **self._params["export"])
        export_qonnx(model, (x,), output_onnx_path, opset_version=14, do_constant_folding=True)

    def _step_build_setup(self):
        """Set up the dataflow build for transformer models with custom attention steps."""
        # with open("params.yaml") as file:
        #    params = yaml.safe_load(file)
        # Seed all RNGs
        seed(self._params["seed"])
        # Extract sequence length and embedding dimension from parameters
        if "model_seq_len" in self._params and "model_emb_dim" in self._params:
            # for dummy Transformer DUT
            seq_len, emb_dim = self._params["model_seq_len"], self._params["model_emb_dim"]
        else:
            # for real input models
            inp_shape = np.load(self._build_inputs["input_npy_path"]).shape
            if len(inp_shape) == 3:
                # for RadioML Transformers
                _, seq_len, emb_dim = inp_shape
            else:
                # for GPTs (why is this different?)
                model = ModelWrapper(self._build_inputs["onnx_path"])
                _, seq_len, emb_dim = model.get_tensor_shape(
                    "/emb_add/input_quant/export_handler/Quant_output_0"
                )

        # Prepare config files
        # TODO: make configurable
        # TODO: log intermediate files such as inp.npy, folding.yaml,
        # or specialize_layers.jon as artifacts, maybe create in unique temp dirs
        specialize_layers_dict = {
            "Defaults": {"preferred_impl_style": ["rtl", ["MVAU", "Thresholding"]]},
            "": {"preferred_impl_style": ""},
        }
        with open("specialize_layers.json", "w") as f:
            json.dump(specialize_layers_dict, f, indent=2)
        with open("folding.yaml", "w") as f:
            f.write(template_folding_yaml)

        # Create a configuration for building the scaled dot-product attention
        # operator to a hardware accelerator
        cfg = build_cfg.DataflowBuildConfig(
            folding_config_file="folding.yaml",
            specialize_layers_config_file="specialize_layers.json",
            standalone_thresholds=True,
            max_multithreshold_bit_width=16,
            mvau_wwidth_max=2048,
            verify_steps=[
                # In this custom flow, VerificationStepType is repurposed as follows:
                # Enables "tidied_up_python" (before lowering) verification in step_prepare_graph
                build_cfg.VerificationStepType.TIDY_UP_PYTHON,
                # Enables "lowered_python" (after lowering) and "prepared_graph_python"
                # (after QONNX to FINN conversion) verification in step_prepare_graph
                build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
                # Enables "streamlined_python" verification after custom step_streamline
                build_cfg.VerificationStepType.STREAMLINED_PYTHON,
                # Enables "folded_hls_cppsim" verification after custom step_apply_folding_config
                build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
                # Enables RTL simulation of default steps contained in the flow:
                build_cfg.VerificationStepType.NODE_BY_NODE_RTLSIM,  # after step_hw_ipgen
                build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,  # after step_create_stitched_ip
            ],
            # File with test inputs for verification
            verify_input_npy=self._build_inputs["input_npy_path"],
            # File with expected test outputs for verification
            verify_expected_output_npy=self._build_inputs["output_npy_path"],
            # Build steps to execute
            steps=[
                # Prepares the QONNX graph to be consumed by FINN: Cleanup, lowering
                # and Quant to MultiThreshold conversion
                prepare_graph(),
                # Unified exhaustive streamlining of complex model topologies
                # including attention, residuals and splits
                step_streamline,
                # conversion of the scaled dot-product attention pattern to
                # hardware, including cleanup and data layout squeezing
                step_convert_attention_to_hw,
                # Convert the elementwise binary operations to hardware operators.
                # These include for example adding residual branches and positional encoding
                step_convert_elementwise_binary_to_hw,
                # Convert Lookup layers, e.g., token embedding, to hardware custom operators
                step_convert_lookup_to_hw,
                # Convert Split and Concat operators to hardware, e.g., splits
                # contained in the GLU activation
                step_convert_split_concat_to_hw,
                # Convert depth-wise convolution MatMuls to VVUs
                step_convert_depth_wise_to_hw,
                # Properly replicate the stream feeding the query, key and value projections
                step_replicate_streams,
                # Convert most other layers supported by FINN to HW operators
                "step_convert_to_hw",
                # Specialize HW layer implementations as either HLS or RTL
                "step_specialize_layers",
                "step_create_dataflow_partition",
                # Set the folding configuration to meet the cycles per sequence target
                set_target_parallelization(seq_len, emb_dim),
                # Apply folding configuration, specifying hardware implementation details
                step_apply_folding_config,
                "step_minimize_bit_width",
                # The ScaledDotProductAttention custom op does not define any estimates
                "step_generate_estimate_reports",
                "step_hw_codegen",
                "step_hw_ipgen",
                "step_create_stitched_ip",
                "step_measure_rtlsim_performance",
                "step_out_of_context_synthesis",  # for synthesis results (e.g. utilization)
                "step_synthesize_bitfile",
                "step_make_driver",
                "step_deployment_package",
            ],
        )

        # TESTING custom vs live FIFO-sizing
        if self._params.get("live_fifo_sizing"):
            # insert default FIFO-sizing step (behind step_generate_estimate_reports)
            for i in range(len(cfg.steps)):
                if cfg.steps[i] == "step_generate_estimate_reports":
                    cfg.steps.insert(i + 1, "step_set_fifo_depths")
            # disable verification if live FIFO-sizing is on
            cfg.verify_steps = None
        else:
            # insert Christoph's custom FIFO-sizing step (behind step_hw_ipgen)
            for i in range(len(cfg.steps)):
                if cfg.steps[i] == "step_hw_ipgen":
                    cfg.steps.insert(
                        i + 1, set_fifo_depths(seq_len, emb_dim, uram_threshold=seq_len)
                    )
        return cfg
