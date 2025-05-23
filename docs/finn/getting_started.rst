.. _getting_started:

***************
Getting Started
***************

Quickstart
==========

1. Install Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
2. Set up ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` environment variables pointing respectively to the Xilinx tools installation directory and version (e.g. ``FINN_XILINX_PATH=/opt/Xilinx`` and ``FINN_XILINX_VERSION=2022.2``)
3. Clone the FINN compiler from the repo: ``git clone https://github.com/Xilinx/finn/`` and go into the directory where it is cloned
4. Execute ``./run-docker.sh quicktest`` to verify your installation.
5. Optionally, follow the instructions on :ref:`PYNQ board first-time setup` or :ref:`Alveo first-time setup` for board setup.
6. Optionally, set up a `Vivado/Vitis license`_.
7. All done! See :ref:`Running FINN in Docker` for the various options on how to run the FINN compiler.


How do I use FINN?
==================

We strongly recommend that you first watch one of the pre-recorded `FINN tutorial <https://www.youtube.com/watch?v=zw2aG4PhzmA&amp%3Bindex=2>`_
videos, then follow the Jupyter notebook tutorials for `training and deploying an MLP for network intrusion detection <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/cybersecurity>`_ .
You may also want to check out the other :ref:`tutorials`, and the `FINN examples repository <https://github.com/Xilinx/finn-examples>`_ .

Our aim in FINN is *not* to accelerate common off-the-shelf neural networks, but instead provide you with a set of tools
to train *customized* networks and create highly-efficient FPGA implementations from them.
In general, the approach for using the FINN framework is as follows:

1. Train your own quantized neural network (QNN) in `Brevitas <https://github.com/Xilinx/brevitas>`_. We have some `guidelines <https://bit.ly/finn-hls4ml-qat-guidelines>`_ on quantization-aware training (QAT).
2. Export to QONNX and convert to FINN-ONNX by following `this tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb>`_ .
3. Use FINN's ``build_dataflow`` system on the exported model by following this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb>`_ or for advanced settings have a look at this `tutorial <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb>`_ .
4. Adjust your QNN topology, quantization settings and ``build_dataflow`` configuration to get the desired results.

Please note that the framework is still under development, and how well this works will depend on how similar your custom network is to the examples we provide.
If there are substantial differences, you will most likely have to write your own
Python scripts that call the appropriate FINN compiler
functions that process your design correctly, or adding new functions (including
Vitis HLS layers)
as required.
The `advanced FINN tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/advanced>`_ can be useful here.
For custom networks, we recommend making a copy of the `BNN-PYNQ end-to-end
Jupyter notebook tutorials <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example/bnn-pynq>`_ as a starting point, visualizing the model at intermediate
steps and adding calls to new transformations as needed.
Once you have a working flow, you can implement a command line entry for this
by using the "advanced mode" described in the :ref:`command_line` section.

Running FINN in Docker
======================
FINN runs inside a Docker container, it comes with a script to easily build and launch the container. If you are not familiar with Docker, there are many excellent `online resources <https://docker-curriculum.com/>`_ to get started.
You may want to review the :ref:`General FINN Docker tips` and :ref:`Environment variables` as well.

The above mentioned script to build and launch the FINN docker container is called `run-docker.sh <https://github.com/Xilinx/finn/blob/main/run-docker.sh>`_ . It can be launched in the following modes:

Launch interactive shell
************************
Simply running bash run-docker.sh without any additional arguments will create a Docker container with all dependencies and give you a terminal with you can use for development for experimentation:

::

  bash ./run-docker.sh


Launch a Build with ``build_dataflow``
**************************************
FINN is currently more compiler infrastructure than compiler, but we do offer
a :ref:`command_line` entry for certain use-cases. These run a predefined flow
or a user-defined flow from the command line as follows:

::

  bash ./run-docker.sh build_dataflow <path/to/dataflow_build_dir/>
  bash ./run-docker.sh build_custom <path/to/custom_build_dir/>


Launch Jupyter notebooks
************************
FINN comes with numerous Jupyter notebook tutorials, which you can launch with:

::

  bash ./run-docker.sh notebook

This will launch the `Jupyter notebook <https://jupyter.org/>`_ server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones.

.. note::
  The link will look something like this (the token you get will be different):
  http://127.0.0.1:8888/?token=f5c6bd32ae93ec103a88152214baedff4ce1850d81065bfc.
  The ``run-docker.sh`` script forwards ports 8888 for Jupyter and 8081 for Netron, and launches the notebook server with appropriate arguments.


Environment variables
**********************

Prior to running the ``run-docker.sh`` script, there are several environment variables you can set to configure certain aspects of FINN.
For a complete list, please have a look in the `run-docker.sh <https://github.com/Xilinx/finn/blob/main/run-docker.sh#L72>`_ file.
The most relevant are summarized below:

* (required) ``FINN_XILINX_PATH`` points to your Xilinx tools installation on the host (e.g. ``/opt/Xilinx``)
* (required) ``FINN_XILINX_VERSION`` sets the Xilinx tools version to be used (e.g. ``2022.2``)
* (required for Alveo) ``PLATFORM_REPO_PATHS`` points to the Vitis platform files (DSA).
* (required for Alveo) ``XRT_DEB_VERSION`` specifies the .deb to be installed for XRT inside the container (see default value in ``run-docker.sh``).
* (optional) ``NUM_DEFAULT_WORKERS`` (default 4) specifies the degree of parallelization for the transformations that can be run in parallel, potentially reducing build time
* (optional) ``FINN_HOST_BUILD_DIR`` specifies which directory on the host will be used as the build directory. Defaults to ``/tmp/finn_dev_<username>``
* (optional) ``JUPYTER_PORT`` (default 8888) changes the port for Jupyter inside Docker
* (optional) ``JUPYTER_PASSWD_HASH`` (default "") Set the Jupyter notebook password hash. If set to empty string, token authentication will be used (token printed in terminal on launch).
* (optional) ``LOCALHOST_URL`` (default localhost) sets the base URL for accessing e.g. Netron from inside the container. Useful when running FINN remotely.
* (optional) ``NETRON_PORT`` (default 8081) changes the port for Netron inside Docker
* (optional) ``IMAGENET_VAL_PATH`` specifies the path to the ImageNet validation directory for tests.
* (optional) ``FINN_DOCKER_TAG`` (autogenerated) specifies the Docker image tag to use.
* (optional) ``FINN_DOCKER_RUN_AS_ROOT`` (default 0) if set to 1 then run Docker container as root, default is the current user.
* (optional) ``FINN_DOCKER_EXTRA`` (default "") pass extra arguments to the ``docker run`` command when executing ``./run-docker.sh``
* (optional) ``FINN_SKIP_DEP_REPOS`` (default "0") skips the download of FINN dependency repos (uses the ones already downloaded under deps/.
* (optional) ``DOCKER_BUILDKIT`` (default "1") enables `Docker BuildKit <https://docs.docker.com/develop/develop-images/build_enhancements/>`_ for faster Docker image rebuilding (recommended).
* (optional) ``FINN_SINGULARITY`` (default "") points to a pre-built Singularity image to use instead of the Docker image. Singularity support is experimental and intended only for systems where Docker is unavailable.

General FINN Docker tips
************************
* Several folders including the root directory of the FINN compiler and the ``FINN_HOST_BUILD_DIR`` will be mounted into the Docker container and can be used to exchange files.
* Do not use ``sudo`` to launch the FINN Docker. Instead, setup Docker to run `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_.
* If you want a new terminal on an already-running container, you can do this with ``docker exec -it <name_of_container> bash``.
* The container is spawned with the `--rm` option, so make sure that any important files you created inside the container are either in the finn compiler folder (which is mounted from the host computer) or otherwise backed up.

Supported FPGA Hardware
=======================
**Vivado IPI support for any Xilinx FPGA:** FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx-AMD FPGA as part of a larger system. It’s up to you to take the FINN-generated accelerator (what we call “stitched IP” in the tutorials), wire it up to your FPGA design and send/receive neural network data to/from the accelerator.

**Shell-integrated accelerator + driver:** For quick deployment, we target boards supported by  `PYNQ <http://www.pynq.io/>`_ . For these platforms, we can build a full bitfile including DMAs to move data into and out of the FINN-generated accelerator, as well as a Python driver to launch the accelerator. We support the Pynq-Z1, Pynq-Z2, Kria SOM, Ultra96, ZCU102 and ZCU104 boards, as well as Alveo cards.

PYNQ board first-time setup
****************************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PYNQ board. To be able to access the target from the host, you'll need to set up SSH public key authentication:

Start on the target side:

1. Note down the IP address of your PYNQ board. This IP address must be accessible from the host.
2. Ensure the ``bitstring`` package is installed: ``sudo pip3 install bitstring``

Continue on the host side (replace the ``<PYNQ_IP>`` and ``<PYNQ_USERNAME>`` with the IP address and username of your board from the first step):

1. Launch the Docker container from where you cloned finn with ``./run-docker.sh``
2. Go into the `ssh_keys` directory  (e.g. ``cd /path/to/finn/ssh_keys``)
3. Run ``ssh-keygen`` to create a key pair e.g. ``id_rsa`` private and ``id_rsa.pub`` public key
4. Run ``ssh-copy-id -i id_rsa.pub <PYNQ_USERNAME>@<PYNQ_IP>`` to install the keys on the remote system
5. Test that you can ``ssh <PYNQ_USERNAME>@<PYNQ_IP>`` without having to enter the password. Pass the ``-v`` flag to the ssh command if it doesn't work to help you debug.


Alveo first-time setup
**********************
We use *host* to refer to the PC running the FINN Docker environment, which will build the accelerator+driver and package it up, and *target* to refer to the PC where the Alveo card is installed. These two can be the same PC, or connected over the network -- FINN includes some utilities to make it easier to test on remote PCs too. Prior to first usage, you need to set up both the host and the target in the following manner:

On the target side:

1. Install Xilinx XRT.
2. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation, for instance ``/opt/xilinx/platforms``.
3. Create a conda environment named *finn-pynq-alveo* by following this guide `to set up PYNQ for Alveo <https://pynq.readthedocs.io/en/latest/getting_started/alveo_getting_started.html>`_. It's best to follow the recommended environment.yml (set of package versions) in this guide.
4. Activate the environment with `conda activate finn-pynq-alveo` and install the bitstring package with ``pip install bitstring``.
5. Done! You should now be able to e.g. ``import pynq`` in Python scripts.



On the host side:

1. Install Vitis 2022.2 and set up the ``VITIS_PATH`` environment variable to point to your installation.
2. Install Xilinx XRT. Ensure that the ``XRT_DEB_VERSION`` environment variable reflects which version of XRT you have installed.
3. Install the Vitis platform files for Alveo and set up the ``PLATFORM_REPO_PATHS`` environment variable to point to your installation. *This must be the same path as the target's platform files (target step 2)*
5. `Set up public key authentication <https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server>`_. Copy your private key to the ``finn/ssh_keys`` folder on the host to get password-less deployment and remote execution.
6. Done!

Vivado/Vitis license
*********************
If you are targeting Xilinx FPGA parts that needs specific licenses (non-WebPack) you can make these available to the
FINN Docker container by passing extra arguments. To do this, you can use the ``FINN_DOCKER_EXTRA`` environment variable as follows:

::

  export FINN_DOCKER_EXTRA=" -v /path/to/licenses:/path/to/licenses -e XILINXD_LICENSE_FILE=/path/to/licenses "

The above example mounts ``/path/to/licenses`` from the host into the same path on the Docker container, and sets the
value of the ``XILINXD_LICENSE_FILE`` environment variable.

System Requirements
====================

* Ubuntu 18.04 with ``bash`` installed
* Docker `without root <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_
* A working Vitis/Vivado 2022.2 installation
* ``FINN_XILINX_PATH`` and ``FINN_XILINX_VERSION`` environment variables correctly set, see `Quickstart`_
* *(optional)* `Vivado/Vitis license`_ if targeting non-WebPack FPGA parts.
* *(optional)* A PYNQ board with a network connection, see `PYNQ board first-time setup`_

We also recommend running the FINN compiler on a system with sufficiently
strong hardware:

* **RAM.** Depending on your target FPGA platform, your system must have sufficient RAM to be
  able to run Vivado/Vitis synthesis for that part. See `this page <https://www.xilinx.com/products/design-tools/vivado/vivado-ml.html#memory>`_
  for more information. For targeting Zynq and Zynq UltraScale+ parts, at least 8 GB is recommended. Larger parts may require up to 16 GB.
  For targeting Alveo parts with Vitis, at least 64 GB RAM is recommended.

* **CPU.** FINN can parallelize HLS synthesis and several other operations for different
  layers, so using a multi-core CPU is recommended. However, this should be balanced
  against the memory usage as a high degree of parallelization will require more
  memory. See the ``NUM_DEFAULT_WORKERS`` environment variable below for more on
  how to control the degree of parallelization.

* **Storage.** While going through the build steps, FINN will generate many files as part of
  the process. For larger networks, you may need 10s of GB of space for the temporary
  files generated during the build.
  By default, these generated files will be placed under ``/tmp/finn_dev_<username>``.
  You can override this location by using the ``FINN_HOST_BUILD_DIR`` environment
  variable.
  Mapping the generated file dir to a fast SSD will result in quicker builds.
