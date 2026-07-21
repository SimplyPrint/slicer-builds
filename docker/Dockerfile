# syntax=docker/dockerfile:1

ARG UBUNTU_IMAGE=ubuntu:24.04
FROM ${UBUNTU_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    GIT_TERMINAL_PROMPT=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CONAN_HOME=/cache/conan \
    CCACHE_DIR=/cache/ccache \
    SCCACHE_DIR=/cache/sccache \
    CMAKE_GENERATOR=Ninja \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# This is intentionally a union of the upstream build prerequisites. Runtime
# dependencies are supplied by the much smaller cloud-slicer runtime image and
# are not copied into each slicer bundle.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        bc \
        bison \
        build-essential \
        ca-certificates \
        ccache \
        clang \
        cmake \
        curl \
        eglexternalplatform-dev \
        extra-cmake-modules \
        file \
        flex \
        gettext \
        git \
        gperf \
        libcairo2-dev \
        libcurl4-openssl-dev \
        libdbus-1-dev \
        libegl1-mesa-dev \
        libexpat1-dev \
        libglew-dev \
        libglu1-mesa-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libhidapi-dev \
        libjpeg-dev \
        libosmesa6-dev \
        libpng-dev \
        libsecret-1-dev \
        libssl-dev \
        libtool \
        libudev-dev \
        libwayland-dev \
        libwebkit2gtk-4.1-dev \
        libxkbcommon-dev \
        lld \
        m4 \
        nasm \
        ninja-build \
        patch \
        perl \
        pkg-config \
        python3 \
        python3-numpy \
        python3-pip \
        python3-venv \
        sccache \
        texinfo \
        unzip \
        wayland-protocols \
        wget \
        xz-utils \
        zip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/conan \
    && /opt/conan/bin/pip install "conan==2.15.1"

ENV PATH=/opt/conan/bin:${PATH}
WORKDIR /workspace/repo
