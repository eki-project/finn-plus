#!/bin/bash

sudo pip install bitstring
sudo pip install finn-dataset-loading
sudo pip install onnx==1.17.0
sudo pip install qonnx==1.0.0
sudo pip install h5py
# Ensure other installs don't upgrade numpy too far:
sudo pip install "numpy<2.0.0"
# Not pulled in automatically by the installs above:
sudo pip install grpcio==1.64.0
# Only needed on boards with older PYNQ images (< 3.1.1) - uncomment if required.
# Workaround for https://discuss.pynq.io/t/how-to-address-axilite-interface-in-pynq-v3-0/4831
#sudo pip install pynqmetadata==0.1.5
