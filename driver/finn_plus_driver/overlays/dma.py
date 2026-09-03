"""Standard DMA-based driver for FINN-generated accelerators."""

import json
import numpy as np
import os
import time
from finn_plus_driver.hwh import get_clk_wiz_params_from_hwh
from finn_plus_driver.packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy
from pynq import Overlay, allocate
from pynq.ps import Clocks
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor


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
        self.fclk_mhz = fclk_mhz  # currently ignored (TODO: make clocking wizard configurable)
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
            clk_wiz_params = get_clk_wiz_params_from_hwh(bitfile_name)
            Clocks.fclk0_mhz = 100.0  # Clocking Wizard is configured for fixed 100 MHz input clock
            self.fclk_mhz_actual = float(
                clk_wiz_params.get(
                    "CLKOUT1_OUT_FREQ",
                    clk_wiz_params.get("CLKOUT1_REQUESTED_OUT_FREQ", str(self.fclk_mhz)),
                )
            )
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
                "weight tensors available do not match. \n"
                "Is runtime_weight_dir pointing to the correct folder?"
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
                with open(self.runtime_weight_dir + "/" + w_filename) as f:
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
        """Block until all output DMAs have finished writing."""
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
        if type(input_npy) is not list:
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
        from finn_plus_driver.validate import run_validate

        validation_dataset = kwargs.get("validation_dataset", self.validation_dataset)
        run_validate(validation_dataset, self, *args, **kwargs)

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
