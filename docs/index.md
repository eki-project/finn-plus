# FINN Framework Documentation

Welcome to the comprehensive documentation for the FINN (Fast, Scalable Quantized Neural Network Inference) framework.

## What is FINN?

FINN is an experimental framework from AMD that generates efficient FPGA accelerators for quantized neural networks. It provides a complete toolchain for generating custom FPGA implementations of quantized neural networks.

## Documentation Sections

### [API Reference](api/)

Comprehensive reference documentation for all FINN modules, classes, and functions:

- **Builder**: Configuration and build system components
- **Utilities**: Helper functions and common utilities
- **Core**: Analysis, transformation, and custom operation modules
- **Custom Operations**: FINN-specific neural network operations

### Key Features

- **Quantized Neural Networks**: Support for low-precision neural networks
- **FPGA Acceleration**: Generate efficient FPGA implementations
- **Dataflow Architecture**: Streaming dataflow accelerators
- **Customizable**: Flexible configuration and optimization options

## Getting Started

To use FINN, you'll typically work with:

1. **Build Configuration**: Use `DataflowBuildConfig` to configure your build
2. **Model Preparation**: Convert and optimize your neural network model
3. **Hardware Generation**: Generate FPGA accelerator implementations
4. **Deployment**: Deploy to target FPGA platforms

## Code Examples

### Basic Build Configuration

```python
from finn.builder.build_dataflow_config import DataflowBuildConfig

# Create build configuration
config = DataflowBuildConfig(
    output_dir="./output",
    synth_clk_period_ns=10.0,
    board="Pynq-Z1",
    target_fps=1000
)
```

### Using Utilities

```python
from finn.util.basic import get_dsp_block

# Get DSP block type for FPGA part
dsp_type = get_dsp_block("xc7z020clg400-1")
print(f"DSP block type: {dsp_type}")  # Output: DSP48E1
```

## Contributing

This documentation is automatically generated from the source code. To improve it:

1. Add or update docstrings in the source code
2. Follow the established docstring conventions (Google style)
3. Include type hints where appropriate

## More Information

- [FINN GitHub Repository](https://github.com/Xilinx/finn)
- [FINN Documentation](https://finn.readthedocs.io/)
- [PYNQ Framework](http://www.pynq.io/)

---

*This documentation is automatically generated and updated when code changes are pushed to the repository.*
