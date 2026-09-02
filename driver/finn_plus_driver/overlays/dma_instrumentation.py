"""Driver combining DMA data movement with instrumentation."""

from finn_plus_driver.overlays.dma import FINNDMAOverlay
from finn_plus_driver.overlays.instrumentation import FINNInstrumentationOverlay


class FINNDMAInstrumentationOverlay(FINNDMAOverlay, FINNInstrumentationOverlay):
    """FINN overlay for DMA and instrumentation (with Switch Block)."""

    def __init__(
        self,
        bitfile_name,
        io_shape_dict,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        runtime_weight_dir="runtime_weights/",
        validation_dataset=None,
        batch_size=1,
        seed=1,
        **kwargs,
    ):
        """Initialize DMA instrumentation overlay."""
        super().__init__(
            bitfile_name,
            io_shape_dict=io_shape_dict,
            platform=platform,
            fclk_mhz=fclk_mhz,
            device=device,
            download=download,
            runtime_weight_dir=runtime_weight_dir,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            seed=seed,
        )

    def set_current_mode(self, mode):
        """Set accelerator mode ('dma' or 'instr')."""
        if self.get_current_mode() != mode:
            self.reset_accelerator()
            val = 1 if mode == "instr" else 0
            self.axi_gpio_0.write(
                offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO2_DATA"]["address_offset"],
                value=val,
            )

    def get_current_mode(self):
        """Get accelerator mode."""
        val = self.axi_gpio_0.read(
            offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO2_DATA"]["address_offset"]
        )
        return "instr" if val == 1 else "dma"

    def throughput_test(self, **kwargs):
        """Run throughput test (DMA mode)."""
        self.set_current_mode("dma")
        return super().throughput_test(**kwargs)

    def execute(self, input_npy):
        """Execute (DMA mode)."""
        self.set_current_mode("dma")
        return super().execute(input_npy)

    def experiment_instrumentation(self, **kwargs):
        """Run instrumentation experiment (instrumentation mode)."""
        self.set_current_mode("instr")
        return super().experiment_instrumentation(**kwargs)

    def validate(self, *args, **kwargs):
        """Run validation in DMA mode."""
        self.set_current_mode("dma")
        return super().validate(*args, **kwargs)
