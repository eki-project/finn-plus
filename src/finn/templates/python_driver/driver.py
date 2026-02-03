"""FINN driver for PYNQ."""

import click
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# from pynq import PL
from pynq import Overlay, allocate

# from pynq.pl_server.device import Device
from pynq.ps import Clocks
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy


class FINNDMAOverlay(Overlay):
    """FINN overlay for DMA."""

    def __init__(
        self,
        bitfile_name,
        platform,
        io_shape_dict,
        batch_size=1,
        fclk_mhz=100.0,
        device=None,
        download=True,
        runtime_weight_dir="runtime_weights/",
        validation_dataset=None,
        **kwargs,
    ):
        """Initialize the FINN accelerator.

        Parameters
        ----------
        bitfile_name: str
            Path to accelerator .bit/.xclbin file
        platform: str
            FINN platform type, either "alveo" or "zynq-iodma"
        io_shape_dict: dict
            Dictionary with particulars of the generated accelerator
        batch_size: int
            Maximum batch size in driver (hardware batchsize is always 1)
        fclk_mhz: float
            Override the clock frequency, only possible for Zynq.
        device: pynq.Device
            Which PYNQ device to use, None for default.
        download: bool
            Whether to flash the bitstream.
        runtime_weight_dir: str
            Path to runtime weights folder.
        """
        super().__init__(bitfile_name, download=download, device=device)
        self.runtime_weight_dir = runtime_weight_dir
        self.io_shape_dict = io_shape_dict
        self.ibuf_packed_device = None
        self.obuf_packed_device = None
        self.platform = platform
        self.batch_size = batch_size
        self.fclk_mhz = fclk_mhz
        self.validation_dataset = validation_dataset
        self.idma = []
        self.odma = []
        self.odma_handle = []
        if "idma_names" in io_shape_dict.keys():
            for idma_name in io_shape_dict["idma_names"]:
                self.idma.append(getattr(self, idma_name))
        else:
            self.idma = [self.idma0]
        if "odma_names" in io_shape_dict.keys():
            for odma_name in io_shape_dict["odma_names"]:
                self.odma.append(getattr(self, odma_name))
                if self.platform == "alveo":
                    self.odma_handle.append(None)
        else:
            self.odma = [self.odma0]
            if self.platform == "alveo":
                self.odma_handle.append(None)
        if self.platform == "zynq-iodma":
            # set the clock frequency as specified by user during transformations
            if self.fclk_mhz > 0:
                Clocks.fclk0_mhz = self.fclk_mhz
        # load any external + runtime weights
        self.load_external_weights()
        self.load_runtime_weights()

    def load_external_weights(self):
        """Load any existing external (DRAM) weights from the specified dir into the
        appropriate layer of the accelerator. Note that this must be enabled
        during the accelerator build process. The weights directory
        is specified as the class member ``runtime_weight_dir``. External (DRAM)
        weights are one .npy file per layer.
        """

        self.external_weights = []
        w_filenames = []
        if not os.path.isdir(self.runtime_weight_dir):
            return
        for dirpath, dirnames, filenames in os.walk(self.runtime_weight_dir):
            w_filenames.extend(filenames)

        tmp_weight_dict = {}

        for w_filename in w_filenames:
            if w_filename.endswith(".npy"):
                weight_tensor = np.load(self.runtime_weight_dir + "/" + w_filename)
            else:
                continue

            idma_name = w_filename.split(".")[0]
            tmp_weight_dict[idma_name] = weight_tensor

        for idma_name in tmp_weight_dict.keys():
            if idma_name in self.ip_dict.keys():
                iwdma = getattr(self, idma_name)
                weight_tensor = tmp_weight_dict[idma_name]
                weight_buf = allocate(weight_tensor.shape, dtype=np.uint8)
                weight_buf[:] = weight_tensor
                # weight_buf.sync_to_device()
                weight_buf.flush()

                input_shape = self._io_shape_dict["external_weights_input_shapes"][idma_name]
                # NHWC input?
                if len(input_shape) == 4:
                    num_repeats = input_shape[1] * input_shape[2]
                else:
                    num_repeats = 1
                self.external_weights += [(iwdma, weight_buf, idma_name, num_repeats)]

        if "number_of_external_weights" in self.io_shape_dict:
            hw_ext_weights = self.io_shape_dict["number_of_external_weights"]
            assert len(self.external_weights) == hw_ext_weights, (
                "Number of hardware external weights and number of external "
                + "weight tensors available do not match. \n"
                + "Is runtime_weight_dir pointing to the correct folder?"
            )

    def load_runtime_weights(self, flush_accel=True, verify=True):
        """Load any existing runtime-writable weights from the specified dir into the
        appropriate layer of the accelerator. Note that this must be enabled
        during the accelerator build process. The runtime weights directory
        is specified as the class member ``runtime_weight_dir``. Runtime-writable
        weights are provided as one .dat file per layer.

        Parameters
        ----------
        flush_accel: bool
            Run the accelerator with dummy input after weights are written to
            flush any stale weight data in the weight streamer FIFOs.
        verify: bool
            Whether the written weights will be re-read and verified.
        """
        w_filenames = []
        if not os.path.isdir(self.runtime_weight_dir):
            return
        for dirpath, dirnames, filenames in os.walk(self.runtime_weight_dir):
            w_filenames.extend(filenames)
        rt_weight_dict = {}
        for w_filename in w_filenames:
            if w_filename.endswith(".dat"):
                with open(self.runtime_weight_dir + "/" + w_filename, "r") as f:
                    dat = f.read()
            else:
                continue
            layer_w = np.fromiter([int(x, 16) for x in dat.strip().split()], dtype=np.uint32)
            sdp_ind = int(w_filename.split("_")[0])
            layer_ind = int(w_filename.split("_")[1])
            rt_weight_dict[(sdp_ind, layer_ind)] = layer_w
        for sdp_ind, layer_ind in rt_weight_dict.keys():
            cand_if_name = "StreamingDataflowPartition_%d" % sdp_ind
            if cand_if_name in self.ip_dict.keys():
                layer_mmio = getattr(self, "StreamingDataflowPartition_%d" % sdp_ind).mmio
                layer_w = rt_weight_dict[(sdp_ind, layer_ind)]
                layer_mmio.write_mm(0, layer_w.tobytes())
                if verify:
                    if self.platform == "alveo":
                        # Pynq for Alveo uses tinynumpy under the hood. There is a bug when going
                        # from a tinynumpy.ndarray to numpy.ndarray. To work around this, we first
                        # convert the tinynumpy.ndarray to a list and then copy the list to a
                        # numpy.ndarray.
                        # There is a known bug with larger sets of weights. Accesses to address
                        # spaces over 16KB do NOT work as intended. Be aware of this if seeing
                        # unexpected behaviour.
                        new_array = layer_mmio.array[: layer_w.shape[0]]
                        new_w = np.copy(np.array(([x for x in new_array]), dtype=layer_w.dtype))
                    else:
                        new_w = np.copy(layer_mmio.array[: layer_w.shape[0]])
                    assert (layer_w == new_w).all()
        if flush_accel:
            # run accelerator to flush any stale weights from weight streamer FIFOs
            self.execute_on_buffers()

    def idt(self, ind=0):
        """Get input data type for specified index."""
        return self.io_shape_dict["idt"][ind]

    def odt(self, ind=0):
        """Get output data type for specified index."""
        return self.io_shape_dict["odt"][ind]

    def ishape_normal(self, ind=0):
        """Get normal input shape with current batch size."""
        ret = list(self.io_shape_dict["ishape_normal"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_normal(self, ind=0):
        """Get normal output shape with current batch size."""
        ret = list(self.io_shape_dict["oshape_normal"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def ishape_folded(self, ind=0):
        """Get folded input shape with current batch size."""
        ret = list(self.io_shape_dict["ishape_folded"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_folded(self, ind=0):
        """Get folded output shape with current batch size."""
        ret = list(self.io_shape_dict["oshape_folded"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def ishape_packed(self, ind=0):
        """Get packed input shape with current batch size."""
        ret = list(self.io_shape_dict["ishape_packed"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    def oshape_packed(self, ind=0):
        """Get packed output shape with current batch size."""
        ret = list(self.io_shape_dict["oshape_packed"][ind])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def num_inputs(self):
        """Number of accelerator inputs."""
        return self.io_shape_dict["num_inputs"]

    @property
    def num_outputs(self):
        """Number of accelerator outputs."""
        return self.io_shape_dict["num_outputs"]

    @property
    def batch_size(self):
        """Current batch size."""
        return self._batch_size

    @property
    def io_shape_dict(self):
        """Dictionary of I/O shapes and data types."""
        return self._io_shape_dict

    @io_shape_dict.setter
    def io_shape_dict(self, value):
        """Set I/O shape dictionary and convert data types."""
        idt = value.get("idt", None)
        if all(isinstance(element, str) for element in idt):
            idt_new = []
            for i in idt:
                type_name = i[i.index("[") + 1 : i.index("]")]
                idt_new.append(DataType[type_name.strip("'")])
            value["idt"] = idt_new

        odt = value.get("odt", None)
        if all(isinstance(element, str) for element in odt):
            odt_new = []
            for o in odt:
                type_name = o[o.index("[") + 1 : o.index("]")]
                odt_new.append(DataType[type_name.strip("'")])
            value["odt"] = odt_new

        self._io_shape_dict = value

    @batch_size.setter
    def batch_size(self, value):
        """Set batch size and reallocate buffers."""
        self._batch_size = value
        # free the old buffers by setting to None
        # (reference counting should care of it)
        if self.ibuf_packed_device is not None:
            self.ibuf_packed_device = None
        if self.obuf_packed_device is not None:
            self.obuf_packed_device = None
        cacheable = {"alveo": False, "zynq-iodma": True}[self.platform]
        self.ibuf_packed_device = []
        self.obuf_packed_device = []
        self.obuf_packed = []
        for i in range(self.num_inputs):
            new_packed_ibuf = allocate(
                shape=self.ishape_packed(i), dtype=np.uint8, cacheable=cacheable, target=self.device
            )
            self.ibuf_packed_device.append(new_packed_ibuf)
        for o in range(self.num_outputs):
            new_packed_obuf = allocate(
                shape=self.oshape_packed(o), dtype=np.uint8, cacheable=cacheable, target=self.device
            )
            self.obuf_packed_device.append(new_packed_obuf)
            self.obuf_packed.append(np.empty_like(new_packed_obuf))

    def fold_input(self, ibuf_normal, ind=0):
        """Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input."""
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal(ind)
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded(ind))
        return ibuf_folded

    def pack_input(self, ibuf_folded, ind=0):
        """Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data."""
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded,
            self.idt(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed, ind=0):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed,
            self.odt(ind),
            self.oshape_folded(ind),
            reverse_endian=True,
            reverse_inner=True,
        )
        return obuf_folded

    def unfold_output(self, obuf_folded, ind=0):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal(ind))
        return obuf_normal

    def copy_input_data_to_device(self, data, ind=0):
        """Copies given input data to PYNQ buffer."""
        np.copyto(self.ibuf_packed_device[ind], data)
        self.ibuf_packed_device[ind].flush()

    def copy_output_data_from_device(self, data, ind=0):
        """Copies PYNQ output buffer from device."""
        self.obuf_packed_device[ind].invalidate()
        np.copyto(data, self.obuf_packed_device[ind])

    def execute_on_buffers(self, asynch=False, batch_size=None):
        """Executes accelerator by setting up the DMA(s) on pre-allocated buffers.
        Blocking behavior depends on the asynch parameter:
        * ``asynch=True`` will block until all transfers are complete.
        * ``asynch=False`` won't block, use ``wait_until_finished()`` to check
           completion

        The optional batch_size parameter can be used to execute on a smaller
        batch than the initialized ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size, "Specified batch_size is too large."
        if self.platform == "zynq-iodma":
            for o in range(self.num_outputs):
                assert self.odma[o].read(0x00) & 0x4 != 0, "Output DMA %d is not idle" % (o)
            # manually launch IODMAs since signatures are missing
            for iwdma, iwbuf, iwdma_name, num_repeats in self.external_weights:
                iwdma.write(0x10, iwbuf.device_address & 0xFFFFFFFF)
                iwdma.write(0x14, (iwbuf.device_address >> 32) & 0xFFFFFFFF)
                iwdma.write(0x1C, batch_size * num_repeats)
                iwdma.write(0x00, 1)
            for o in range(self.num_outputs):
                self.odma[o].write(0x10, self.obuf_packed_device[o].device_address & 0xFFFFFFFF)
                self.odma[o].write(
                    0x14, (self.obuf_packed_device[o].device_address >> 32) & 0xFFFFFFFF
                )
                self.odma[o].write(0x1C, batch_size)
                self.odma[o].write(0x00, 1)
            for i in range(self.num_inputs):
                self.idma[i].write(0x10, self.ibuf_packed_device[i].device_address & 0xFFFFFFFF)
                self.idma[i].write(
                    0x14, (self.ibuf_packed_device[i].device_address >> 32) & 0xFFFFFFFF
                )
                self.idma[i].write(0x1C, batch_size)
                self.idma[i].write(0x00, 1)
        elif self.platform == "alveo":
            for o in range(self.num_outputs):
                assert self.odma_handle[o] is None, "Output DMA %d is already running" % o
            for i in range(self.num_inputs):
                self.idma[i].start(self.ibuf_packed_device[i], batch_size)
            for iwdma, iwbuf, iwdma_name, num_repeats in self.external_weights:
                iwdma.start(iwbuf, batch_size * num_repeats)
            for o in range(self.num_outputs):
                self.odma_handle[o] = self.odma[o].start(self.obuf_packed_device[o], batch_size)
        else:
            raise Exception("Unrecognized platform: %s" % self.platform)
        # blocking behavior depends on asynch parameter
        if asynch is False:
            self.wait_until_finished()

    def wait_until_finished(self):
        "Block until all output DMAs have finished writing."
        if self.platform == "zynq-iodma":
            # check if output IODMA is finished via register reads
            for o in range(self.num_outputs):
                status = self.odma[o].read(0x00)
                while status & 0x2 == 0:
                    status = self.odma[o].read(0x00)
        elif self.platform == "alveo":
            assert all([x is not None for x in self.odma_handle]), "No odma_handle to wait on"
            for o in range(self.num_outputs):
                self.odma_handle[o].wait()
                self.odma_handle[o] = None
        else:
            raise Exception("Unrecognized platform: %s" % self.platform)

    def execute(self, input_npy):
        """Given a single or a list of input numpy array, first perform necessary
        packing and copying to device buffers, execute on accelerator, then unpack
        output and return output numpy array from accelerator."""
        # if single input, convert to list to normalize how we process the input
        if not type(input_npy) is list:
            input_npy = [input_npy]
        assert self.num_inputs == len(input_npy), "Not all accelerator inputs are specified."
        for i in range(self.num_inputs):
            ibuf_folded = self.fold_input(input_npy[i], ind=i)
            ibuf_packed = self.pack_input(ibuf_folded, ind=i)
            self.copy_input_data_to_device(ibuf_packed, ind=i)
        self.execute_on_buffers()
        outputs = []
        for o in range(self.num_outputs):
            self.copy_output_data_from_device(self.obuf_packed[o], ind=o)
            obuf_folded = self.unpack_output(self.obuf_packed[o], ind=o)
            obuf_normal = self.unfold_output(obuf_folded, ind=o)
            outputs.append(obuf_normal)
        if self.num_outputs == 1:
            return outputs[0]
        else:
            return outputs

    def throughput_test(self, **kwargs):
        """Run accelerator with empty inputs to measure throughput and other metrics.
        Returns dictionary with various metrics."""
        # dictionary for results of throughput test
        res = {}
        start = time.time()
        self.execute_on_buffers()
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime * 1000
        res["throughput[images/s]"] = self.batch_size / runtime
        total_in = 0
        for i in range(self.num_inputs):
            total_in += np.prod(self.ishape_packed(i))
        res["DRAM_in_bandwidth[MB/s]"] = total_in * 0.000001 / runtime
        total_out = 0
        for o in range(self.num_outputs):
            total_out += np.prod(self.oshape_packed(o))
        res["DRAM_out_bandwidth[MB/s]"] = total_out * 0.000001 / runtime
        for iwdma, iwbuf, iwdma_name, num_repeats in self.external_weights:
            res["DRAM_extw_%s_bandwidth[MB/s]" % iwdma_name] = (
                self.batch_size * np.prod(iwbuf.shape) * num_repeats * 0.000001 / runtime
            )
        if self.platform == "zynq-iodma":
            res["fclk[mhz]"] = Clocks.fclk0_mhz
        elif self.platform == "alveo":
            res["fclk[mhz]"] = self.clock_dict["clock0"]["frequency"]
        res["batch_size"] = self.batch_size
        # also benchmark driver-related overheads
        input_npy = gen_finn_dt_tensor(self.idt(), self.ishape_normal())
        # provide as int8/uint8 to support fast packing path where possible
        if self.idt() == DataType["UINT8"]:
            input_npy = input_npy.astype(np.uint8)
        elif self.idt() == DataType["INT8"]:
            input_npy = input_npy.astype(np.int8)
        start = time.time()
        ibuf_folded = self.fold_input(input_npy)
        end = time.time()
        runtime = end - start
        res["fold_input[ms]"] = runtime * 1000

        start = time.time()
        ibuf_packed = self.pack_input(ibuf_folded)
        end = time.time()
        runtime = end - start
        res["pack_input[ms]"] = runtime * 1000

        start = time.time()
        self.copy_input_data_to_device(ibuf_packed)
        end = time.time()
        runtime = end - start
        res["copy_input_data_to_device[ms]"] = runtime * 1000

        start = time.time()
        self.copy_output_data_from_device(self.obuf_packed[0])
        end = time.time()
        runtime = end - start
        res["copy_output_data_from_device[ms]"] = runtime * 1000

        start = time.time()
        obuf_folded = self.unpack_output(self.obuf_packed[0])
        end = time.time()
        runtime = end - start
        res["unpack_output[ms]"] = runtime * 1000

        start = time.time()
        self.unfold_output(obuf_folded)
        end = time.time()
        runtime = end - start
        res["unfold_output[ms]"] = runtime * 1000
        return res

    def validate(self, *args, **kwargs):
        """Validate accelerator accuracy on dataset."""
        validation_dataset = kwargs.get("validation_dataset", self.validation_dataset)
        dataset_root = kwargs.get(
            "dataset_root", os.path.join(os.path.dirname(os.path.realpath(__file__)), "validate")
        )

        sys.path.insert(0, dataset_root)
        try:
            import run_validate
        finally:
            # Remove the added path to avoid side effects
            if dataset_root in sys.path:
                sys.path.remove(dataset_root)

        run_validate.run_validate(validation_dataset, self, *args, **kwargs)

    def idle(self, *args, **kwargs):
        """Run idle for specified time."""
        runtime = kwargs.get("time")
        print("Running idle for %d seconds.." % runtime)
        time.sleep(runtime)
        print("Done.")

    def run_throughput_test(self, *args, **kwargs):
        """Run throughput test and save report."""
        report_dir = kwargs.get("report_dir")
        res = self.throughput_test()
        print(res)
        reportfile = os.path.join(report_dir, "report_throughput_test.json")
        with open(reportfile, "w") as f:
            json.dump(res, f, indent=2)


class FINNInstrumentationOverlay(Overlay):
    """FINN overlay for instrumentation."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        **kwargs,
    ):
        """Initialize instrumentation overlay."""
        super().__init__(bitfile_name, download=download, device=device)

        self.platform = platform
        self.fclk_mhz = fclk_mhz
        self.seed = seed

        # configure clock (for ZYNQ platforms)
        if self.platform == "zynq":
            if self.fclk_mhz > 0:
                Clocks.fclk0_mhz = self.fclk_mhz
                self.fclk_mhz_actual = Clocks.fclk0_mhz

    def instrumentation_read(self, name):
        """Read instrumentation register."""
        return self.instrumentation_wrap_0.read(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"]
        )

    def instrumentation_write(self, name, value):
        """Write instrumentation register."""
        return self.instrumentation_wrap_0.write(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"],
            value=value,
        )

    def reset_accelerator(self):
        """Reset the accelerator."""
        self.axi_gpio_0.write(
            offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO_DATA"]["address_offset"], value=0
        )

    def start_accelerator(self):
        """Start the accelerator."""
        lfsr_seed = (self.seed << 16) & 0xFFFF0000  # upper 16 bits
        self.instrumentation_write("cfg", lfsr_seed + 1)  # start operation

    def observe_instrumentation(self, debug_print=True):
        """Read and report instrumentation metrics."""
        status_reg = self.instrumentation_read("status")
        chksum_reg = self.instrumentation_read("checksum")
        min_latency = self.instrumentation_read("min_latency")
        latency = self.instrumentation_read("latency")
        interval = self.instrumentation_read("interval")

        frame = (chksum_reg >> 24) & 0x000000FF
        checksum = chksum_reg & 0x00FFFFFF
        overflow_err = (status_reg & 0x00000001) != 0
        underflow_err = (status_reg & 0x00000002) != 0

        if debug_print:
            print("---INSTRUMENTATION_REPORT---")
            if overflow_err or underflow_err:
                print("Status ERROR")
                print("Overflow error: %s" % overflow_err)
                print("Underflow error: %s" % underflow_err)
            else:
                print("Status OK")
            print("Frame number (8-bit): %d" % frame)
            print("Checksum: 0x%06x" % checksum)
            print("Min Latency (cycles): %d" % min_latency)
            print("Latency (cycles): %d" % latency)
            print("Interval (cycles): %d" % interval)
            print("----------------------------")

        return (overflow_err, underflow_err, frame, checksum, min_latency, latency, interval)

    def experiment_instrumentation(self, *args, **kwargs):
        """Run instrumentation experiment and save report."""
        runtime = kwargs.get("runtime")
        report_dir = kwargs.get("report_dir")

        # start accelerator
        print("Running accelerator for %d seconds.." % runtime)
        self.start_accelerator()

        # let it run for specified runtime
        time.sleep(runtime)

        # read measurement from instrumentation
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
        ) = self.observe_instrumentation()

        # write report to file
        report = {
            "error": overflow_err or underflow_err or interval == 0,
            "checksum": checksum,
            "min_latency_cycles": min_latency,
            "latency_cycles": latency,
            "interval_cycles": interval,
            "frequency_mhz": round(self.fclk_mhz_actual),
            "min_latency_ms": round(min_latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "latency_ms": round(latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "throughput_fps": round(1 / (interval * (1 / (self.fclk_mhz_actual * 1e6)))),
            "min_pipeline_depth": round(min_latency / interval, 2),
            "pipeline_depth": round(latency / interval, 2),
        }
        reportfile = os.path.join(report_dir, "report_experiment_instrumentation.json")
        with open(reportfile, "w") as f:
            json.dump(report, f, indent=2)

        print("Done.")

    def idle(self, *args, **kwargs):
        """Run idle for specified time."""
        runtime = kwargs.get("time")
        print("Running idle for %d seconds.." % runtime)
        time.sleep(runtime)
        print("Done.")


class FINNLiveFIFOOverlay(FINNInstrumentationOverlay):
    """FINN overlay for live FIFO sizing."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        fifo_widths=dict(),
        **kwargs,
    ):
        """Initialize live FIFO overlay."""
        super().__init__(
            bitfile_name,
            platform=platform,
            fclk_mhz=fclk_mhz,
            seed=seed,
            download=download,
            device=device,
        )

        self.error = False
        self.fifo_widths = fifo_widths
        self.num_fifos = len(self.fifo_widths)
        # Account for additional FIFO depth and implicit registers introduced by the virtual FIFO
        # HLS implementation that are not present in real FIFOs. This results in a minimum possible
        # FIFO depth of 1 + 8 = 9, which should be improved in a future implementation (TODO).
        self.fifo_depth_offset = 8

        # Sanity check
        # We expect 3 AXI-Lite peripherals next to the virtual FIFOs:
        # instrumentation_wrap_0, axi_gpio_0 (for reset), zynq_ps
        # We expect no additional FINN SDPs with AXI-Lite, such as runtime-writable weights
        if (len(self.ip_dict.keys()) - 3) != self.num_fifos:
            print(
                "Error: # of expected FIFOs (%d) doesn't match # of AXI-Lite interfaces (%d)"
                % (self.num_fifos, len(self.ip_dict.keys()) - 3)
            )
            self.error = True

    def configure_fifo(self, i, mode, depth=2):
        """Configure virtual FIFO mode and depth."""
        # Virtual FIFO register map
        mode_offset = 0x10
        depth_offset = 0x18
        # occupancy_offset = 0x20
        # occupancy_ctrl_offset = 0x24
        # max_occupancy_offset = 0x30
        # max_occupancy_ctrl_offset = 0x34

        ip_name = "StreamingDataflowPartition_%d" % i
        getattr(self, ip_name).write(offset=mode_offset, value=mode)
        getattr(self, ip_name).write(offset=depth_offset, value=depth)

    def total_fifo_size(self, depths):
        """Calculate total FIFO size in kB."""
        # Assuming FIFO SDP/AXI-Lite interfaces are ordered consistently with FIFO IDs
        total_size_bits = 0
        for i, depth in enumerate(depths):
            total_size_bits += (depth + self.fifo_depth_offset) * self.fifo_widths[str(i)]
        total_size_kB = total_size_bits / 8.0 / 1000.0
        return total_size_kB

    def size_iteratively(self, start_depth, iteration_runtime, reduction_factor=0.5):
        """Iteratively reduce FIFO depths to find minimum."""
        fifo_minimum_reached = [False] * self.num_fifos

        if isinstance(start_depth, list):
            # Individual start depth for each FIFO has been supplied
            fifo_depths = start_depth
        else:
            # Initialize all depths to the same start depth
            fifo_depths = [start_depth] * self.num_fifos

        # Reset accelerator and configure FIFOs
        self.reset_accelerator()
        for i in range(0, self.num_fifos):
            self.configure_fifo(i, mode=1, depth=fifo_depths[i])

        # Run once to determine target interval
        self.start_accelerator()
        time.sleep(1)
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
        ) = self.observe_instrumentation(False)
        log_total_fifo_size = [int(self.total_fifo_size(fifo_depths))]
        log_interval = [interval]
        log_min_latency = [min_latency]
        log_latency = [latency]
        target_interval = interval

        # Iteratively reduce FIFO depth until all FIFOs are minimized
        iteration = 0
        start_time = time.time()
        while not all(fifo_minimum_reached):
            for fifo_id in range(0, self.num_fifos):
                if not fifo_minimum_reached[fifo_id]:
                    fifo_depth_before = fifo_depths[fifo_id]
                    fifo_depths[fifo_id] = int(fifo_depths[fifo_id] * reduction_factor)

                    # Reset accelerator
                    self.reset_accelerator()

                    # Configure all FIFOs
                    for i in range(0, self.num_fifos):
                        self.configure_fifo(i, mode=1, depth=fifo_depths[i])

                    # Start accelerator
                    self.start_accelerator()

                    # Let it run
                    time.sleep(iteration_runtime)

                    # Check if throughput dropped or deadlock occured
                    (
                        overflow_err,
                        underflow_err,
                        frame,
                        checksum,
                        min_latency,
                        latency,
                        interval,
                    ) = self.observe_instrumentation(False)

                    if interval > target_interval or interval == 0 or overflow_err or underflow_err:
                        # Revert depth reduction and mark FIFO as minimized
                        fifo_depths[fifo_id] = fifo_depth_before
                        fifo_minimum_reached[fifo_id] = True
                    else:
                        log_total_fifo_size.append(int(self.total_fifo_size(fifo_depths)))
                        log_interval.append(interval)
                        log_min_latency.append(min_latency)
                        log_latency.append(latency)

                    if fifo_depths[fifo_id] == 1:
                        fifo_minimum_reached[fifo_id] = True

            # Report status
            print("Iteration: %d" % iteration)
            print("Numer of minimized FIFOs: %d/%d" % (sum(fifo_minimum_reached), self.num_fifos))
            print("Interval: %d" % log_interval[-1])
            print("Min. latency / latency: %d/%d" % (log_min_latency[-1], log_latency[-1]))
            print("Total FIFO Size (kB): %d" % log_total_fifo_size[-1])

            iteration += 1

        end_time = time.time()
        duration = int(end_time - start_time)
        print("Done (%d seconds)" % duration)

        return (
            fifo_depths,
            log_total_fifo_size,
            log_interval,
            log_min_latency,
            log_latency,
            duration,
        )

    def determine_start_depth(
        self,
    ):
        """
        Attempt to determine start depth for all FIFOs automatically.
        If it doesn't find a working setting start depth must be set manually,
        potentially on per-FIFO basis.
        """
        start_depth = 1
        last_start_depth = 1
        last_interval = 0
        start_depth_found = False

        while not start_depth_found and not self.error:
            print("Testing start depth of %d" % start_depth)
            self.reset_accelerator()

            # Configure FIFOs
            for i in range(0, self.num_fifos):
                self.configure_fifo(i, mode=1, depth=start_depth)

            # Start accelerator and let it run for a long time
            self.start_accelerator()
            time.sleep(1)

            # Examine performance
            (
                overflow_err,
                underflow_err,
                frame,
                checksum,
                min_latency,
                latency,
                interval,
            ) = self.observe_instrumentation()
            if (
                interval > 0
                and interval == last_interval
                and not overflow_err
                and not underflow_err
            ):
                # Accelerator runs with stable interval, reset to previous start depth
                start_depth_found = True
                start_depth = last_start_depth
            else:
                # Start depth is still too small, increase for next try
                last_start_depth = start_depth
                start_depth = start_depth * 2

            last_interval = interval

            if start_depth > 1000000:
                print("Couldn't find a working start depth, please set manually")
                self.error = True

        # Determine runtime per iteration based on performance, so that stable-state is guaranteed
        # Use a simple overestimation for now to be safe
        iteration_runtime = max(0.01, (min_latency * 5) * 10 / 1000 / 1000 / 1000)

        print("Determined start depth for all FIFOs: %d" % start_depth)
        print("Determined iteration runtime based on performance: %f s" % iteration_runtime)
        return (start_depth, iteration_runtime)

    def experiment_fifosizing(self, *args, **kwargs):
        """Run live FIFO sizing experiment and save report."""
        report_dir = kwargs.get("report_dir")

        # For live FIFO-sizing, we also expect the FIFO widths (in bits) exported by FINN, e.g.,
        # {'fifo_widths': {"0": 8, "1": 32, "2": 24}}
        # fifo_widths = kwargs.get("fifo_widths")

        # The settings can also contain the original folding config,
        # into which we can insert the live FIFO sizes once we are done
        folding_config_lfs = kwargs.get("folding_config_before_lfs")

        reportfile = os.path.join(report_dir, "report_experiment_fifosizing.json")
        folding_config_lfs = None

        print("Determining start depth..")
        (start_depth, iteration_runtime) = self.determine_start_depth()

        # First pass
        print("Starting first pass..")
        pass1_result = self.size_iteratively(start_depth, iteration_runtime)
        (
            fifo_depths,
            log_total_fifo_size,
            log_interval,
            log_min_latency,
            log_latency,
            duration,
        ) = pass1_result

        # Visualize results and save as "fifo_sizing_graph.png"
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Total FIFO Size [kB]", color=color)
        ax1.plot(range(len(log_total_fifo_size)), log_total_fifo_size, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(0, max(log_total_fifo_size))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel("Latency [cycles]", color=color)
        ax2.plot(range(len(log_total_fifo_size)), log_latency, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        # ax2.set_ylim(0, max(log_latency))

        ax2.axhline(log_min_latency[0], color="green", label="Minimum (1st frame) Latency")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "fifo_sizing_graph.png"), dpi=300)

        # Second pass for fine-tuning
        print("Starting second pass..")
        pass2_result = self.size_iteratively(fifo_depths, iteration_runtime, reduction_factor=0.95)
        (
            fifo_depths,
            log_total_fifo_size,
            log_interval,
            log_min_latency,
            log_latency,
            duration,
        ) = pass2_result

        # Generate fifo_sizing_report.json
        fifo_report = {
            "error": self.error,
            "fifo_size_total_kB": log_total_fifo_size[-1],
            "fifo_depths": {},
            "fifo_sizes": {},
            "pass_1": {
                "duration": pass1_result[5],
                "log_total_fifo_size": pass1_result[1],
                "log_interval": pass1_result[2],
                "log_min_latency": pass1_result[3],
                "log_latency": pass1_result[4],
            },
            "pass_2": {
                "duration": pass2_result[5],
                "log_total_fifo_size": pass2_result[1],
                "log_interval": pass2_result[2],
                "log_min_latency": pass2_result[3],
                "log_latency": pass2_result[4],
            },
        }
        for fifo, depth in enumerate(fifo_depths):
            size = (depth + self.fifo_depth_offset) * self.fifo_widths[str(fifo)]
            fifo_report["fifo_depths"][fifo] = depth + self.fifo_depth_offset
            fifo_report["fifo_sizes"][fifo] = size
        with open(os.path.join(report_dir, "fifo_sizing_report.json"), "w") as f:
            json.dump(fifo_report, f, indent=2)

        # Generate fifo_depth_export.json to export FIFO depths for use in FINN
        fifo_depth_export = {}
        for fifo, depth in enumerate(fifo_depths):
            fifo_name = "StreamingFIFO_rtl_%d" % fifo
            fifo_depth_export[fifo_name] = {}
            fifo_depth_export[fifo_name]["depth"] = depth + self.fifo_depth_offset
        with open(os.path.join(report_dir, "fifo_depth_export.json"), "w") as f:
            json.dump(fifo_depth_export, f, indent=2)

        # Also export directly into original folding config for convenience
        if folding_config_lfs:
            for key in list(folding_config_lfs.keys()):
                if key.startswith("StreamingFIFO"):
                    fifo_name = "StreamingFIFO_rtl_%d" % int(key.removeprefix("StreamingFIFO_"))
                    # Rename FIFO from StreamingFIFO_* to StreamingFIFO_rtl_*
                    folding_config_lfs[fifo_name] = folding_config_lfs.pop(key)
                    folding_config_lfs[fifo_name]["depth"] = fifo_depth_export[fifo_name]["depth"]
                    folding_config_lfs[fifo_name]["impl_style"] = "rtl"
            with open(os.path.join(report_dir, "folding_config_lfs.json"), "w") as f:
                json.dump(folding_config_lfs, f, indent=2)

        # Generate the usual instrumentation performance report based on final state
        min_latency = log_min_latency[-1]
        latency = log_latency[-1]
        interval = log_interval[-1]
        report = {
            "error": self.error,
            "checksum": 0,
            "min_latency_cycles": min_latency,
            "latency_cycles": latency,
            "interval_cycles": interval,
            "frequency_mhz": round(self.fclk_mhz_actual),
            "min_latency_ms": round(min_latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "latency_ms": round(latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "throughput_fps": round(1 / (interval * (1 / (self.fclk_mhz_actual * 1e6)))),
            "min_pipeline_depth": round(min_latency / interval, 2),
            "pipeline_depth": round(latency / interval, 2),
        }
        with open(reportfile, "w") as f:
            json.dump(report, f, indent=2)

        print("Done.")


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
        self.set_current_mode("dma")
        return super().validate(*args, **kwargs)


def parse_kv(ctx, self, value):
    """Parse key-value pairs from CLI arguments."""
    result = {}
    for item in value:
        if len(item) != 2:
            print(item)
            raise click.UsageError(
                "Items must be in form: key=val TYPE. "
                'With datatypes ["Str", "Int", "Bool", "Float"] being supported'
            )
        if item[0].count("=") != 1:
            raise click.BadParameter("Items must be key=value")
        k, v = item[0].split("=", 1)

        data_type = item[1]
        if data_type == "Str":
            v = v
        elif data_type == "Int":
            v = int(v)
        elif data_type == "Bool":
            # Is always True except for v == False
            v = not (v == "False")
        elif data_type == "Float":
            v = float(v)
        else:
            raise click.BadParameter(
                f'Only datatypes ["Str", "Int", "Bool", "Float"] '
                f"are supported. Used datatype: {data_type}"
            )

        result[k] = v
    return result


@click.command(
    "Example: python driver.py -b ../bitfile/finn-accel.bit "
    "-s ./settings.json -f experiment_instrumentation "
    "-ck seed=42 Int -fk runtime=10 Int "
    "-fk report_dir='./report_dir/' Str"
)
@click.option("--bitfile_name", "-b", help="Path to the Bitstream")
@click.option("--settings", "-s", help="Path to the settings.json")
@click.option("--function", "-f", help="Function to be executed")
@click.option(
    "--ckwarg",
    "-ck",
    multiple=True,
    callback=parse_kv,
    nargs=2,
    help=("Keyword argument for the class instance: " "... -ck key1=val1 TYPE -ck key2=val2 TYPE"),
)
@click.option(
    "--fkwarg",
    "-fk",
    multiple=True,
    callback=parse_kv,
    nargs=2,
    help=("Keyword argument for the called function: " "... -fk key1=val1 TYPE -fk key2=val2 TYPE"),
)
def driver_cli(bitfile_name, settings, function, ckwarg, fkwarg):
    """
    CLI tool to instantiate driver and execute functions.

    Instantiates a driver class and executes a member function.
    The instantiation implicitly loads a bitstream to the FPGA.
    Requires FINN generated bitstream file and settings.json.
    Driver class is inferred from settings.json, while the called
    member function must be chosen via the function option.
    Kwargs for class instantiation or function call can be input
    via --ckwarg or --fkwarg options respectively.
    Class Kwargs take precedence over settings.json Kwargs.
    """

    with open(settings, "r", encoding="utf-8") as f:
        driver_settings = json.load(f)["driver_information"]

    if ckwarg is None:
        ckwarg = {}
    if fkwarg is None:
        fkwarg = {}

    driver_type = driver_settings["driver_type"]
    input_kwargs = {
        **driver_settings,
        **ckwarg,
    }  # ckwarg has precedence when a key conflict happens

    cla = getattr(sys.modules[__name__], driver_type)
    inst = cla(bitfile_name, **input_kwargs)
    func = getattr(inst, function)
    print(func(**fkwarg))


if __name__ == "__main__":
    driver_cli()
