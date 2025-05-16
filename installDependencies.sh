#!/bin/bash

XRT_DEB_VERSION="xrt_202220.2.14.354_22.04-amd64-xrt"

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libc6-dev-i386 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zsh \
    zip \
    perl \
    make \
    autoconf \
    g++ \
    flex \
    bison \
    ccache \
    libgoogle-perftools-dev \
    numactl \
    perl-doc \
    libfl2 \
    libfl-dev \
    zlib1g \
    zlib1g-dev \
    pybind11-dev \
    libfmt-dev \
    libboost-dev \
    libjansson-dev \
    libgetdata-dev \
    g++-10 \
    libpython3.10-dev \
    python3.10-dev

xrt_found=$(dpkg -l | grep xrt | wc -l)
if [ $xrt_found -eq 0 ]; then
    wget -U 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17' "https://www.xilinx.com/bin/public/openDownload?filename=$XRT_DEB_VERSION.deb" -O /tmp/$XRT_DEB_VERSION.deb
    sudo apt install -y /tmp/$XRT_DEB_VERSION.deb
    rm /tmp/$XRT_DEB_VERSION.deb
fi
