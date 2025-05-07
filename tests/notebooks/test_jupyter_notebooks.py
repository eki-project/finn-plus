import pytest

import nbformat
import os
from _pytest.mark.structures import ParameterSet
from nbconvert.preprocessors import ExecutePreprocessor

notebook_timeout_seconds = 3600
notebook_basic_dir = os.path.join(os.environ["FINN_NOTEBOOKS"], "basics/")
notebook_advanced_dir = os.path.join(os.environ["FINN_NOTEBOOKS"], "advanced/")
notebook_cyber_dir = os.path.join(os.environ["FINN_NOTEBOOKS"], "end2end_example/cybersecurity/")
notebook_bnn_dir = os.path.join(os.environ["FINN_NOTEBOOKS"], "end2end_example/bnn-pynq/")

basics_notebooks = [
    pytest.param(
        notebook_basic_dir + "0_how_to_work_with_onnx.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
    pytest.param(
        notebook_basic_dir + "1_brevitas_network_import_via_QONNX.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
]

advanced_notebooks = [
    pytest.param(
        notebook_advanced_dir + "0_custom_analysis_pass.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
    pytest.param(
        notebook_advanced_dir + "1_custom_transformation_pass.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
    pytest.param(
        notebook_advanced_dir + "2_custom_op.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
    pytest.param(
        notebook_advanced_dir + "3_folding.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_general"),
    ),
    pytest.param(
        notebook_advanced_dir + "4_advanced_builder_settings.ipynb",
        marks=[
            pytest.mark.xdist_group(name="notebooks_general"),
            pytest.mark.xfail(reason="Outstanding data layout issue"),
        ],
    ),
]

cyber_notebooks = [
    pytest.param(
        notebook_cyber_dir + "1-train-mlp-with-brevitas.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_cybsec"),
    ),
    pytest.param(
        notebook_cyber_dir + "2-import-into-finn-and-verify.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_cybsec"),
    ),
    pytest.param(
        notebook_cyber_dir + "3-build-accelerator-with-finn.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_cybsec"),
    ),
]

bnn_notebooks = [
    pytest.param(
        notebook_bnn_dir + "cnv_end2end_example.ipynb",
        marks=[
            pytest.mark.xdist_group(name="notebooks_cnv"),
            pytest.mark.xfail(reason="Outstanding data layout issue"),
        ],
    ),
    pytest.param(
        notebook_bnn_dir + "tfc_end2end_example.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_tfc"),
    ),
    pytest.param(
        notebook_bnn_dir + "tfc_end2end_verification.ipynb",
        marks=pytest.mark.xdist_group(name="notebooks_tfc"),
    ),
]


@pytest.mark.notebooks
class Test_notebooks:
    @pytest.mark.parametrize(
        "notebook", basics_notebooks + advanced_notebooks + cyber_notebooks + bnn_notebooks
    )
    def test_notebook_exec(self, notebook: ParameterSet, request):
        with open(notebook) as f:
            # Set different NETRON_PORT for each xdist group to avoid conflicts
            xdist_groups = [
                "notebooks_general",
                "notebooks_cybsec",
                "notebooks_cnv",
                "notebooks_tfc",
            ]
            for mark in request.node.own_markers:
                if mark.name == "xdist_group":
                    group = mark.kwargs["name"]
                    os.environ["NETRON_PORT"] = str(8081 + xdist_groups.index(group))
                    break

            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=notebook_timeout_seconds, kernel_name="python3")
            try:
                assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
            except Exception:
                assert False, f"Failed executing {notebook}"
