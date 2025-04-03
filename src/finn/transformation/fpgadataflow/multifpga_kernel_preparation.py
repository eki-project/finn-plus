import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow.multifpga_network import AuroraNetworkMetadata
from finn.util.basic import make_build_dir
from finn.util.deps import get_deps_path


class PrepareAuroraFlow(Transformation):
    """Use the AuroraNetworkMetadata to package all necessary kernels. Sets the aurora_storage
    metadata prop of the model to point to the directory containing the object files. This does
    nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.aurora_storage = Path(make_build_dir("aurora_storage_")).absolute()
        self.aurora_path = get_deps_path() / "AuroraFlow"
        assert self.aurora_path.exists(), f"Could not find AuroraFlow at {self.aurora_path}"

    def package_single(self, args: str, kernel_xo: str, save_as_xo: str) -> Path:
        """Package a single aurora core and put it into the given location with the given name.
        Copies aurora so that multiple packaging processes can happen at once"""
        temp_dir = Path(make_build_dir("aurora_temp_builddir_"))
        shutil.copytree(self.aurora_path, temp_dir, dirs_exist_ok=True)
        subprocess.run(shlex.split(f"make aurora {args}"), cwd=temp_dir, stdout=subprocess.DEVNULL)
        p_origin = temp_dir / kernel_xo
        assert p_origin.exists(), f"Packaging AuroraFlow failed. Check logs in  {temp_dir}"
        p_target = self.aurora_storage / save_as_xo
        shutil.move(p_origin, p_target)
        assert p_target.exists(), f"Move failed. Target was: {p_target}"
        # We can now safely delete the temp build dir
        shutil.rmtree(temp_dir)
        return p_target.absolute()

    def package_all_from_metadata(self, metadata: AuroraNetworkMetadata) -> None:
        # List all auroras that need to be packaged
        auroras = []
        for device in metadata.table.keys():
            auroras += metadata.get_aurora_kernels(device)

        # Package a single aurora
        def _package_aurora(d: tuple[int, str]) -> None:
            i, aurora_name = d
            origin = f"aurora_flow_{i}.xo"
            target = f"{aurora_name}.xo"
            # TODO: args?
            self.package_single("", origin, target)

        # Package all Aurora kernels concurrently
        with ThreadPoolExecutor(max_workers=int(os.environ["NUM_DEFAULT_WORKERS"])) as tpe:
            tpe.map(_package_aurora, auroras)
            tpe.shutdown()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        metadata = AuroraNetworkMetadata(model)
        model.set_metadata_prop("aurora_storage", str(self.aurora_storage.absolute()))
        self.package_all_from_metadata(metadata)
        return model, False
