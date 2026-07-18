# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH=amd64

RUN test "${TARGETARCH}" = amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        patchelf \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN . slicers/Cura/version.env \
    && git clone --filter=blob:none --no-checkout "https://github.com/${CURA_ENGINE_REPO}.git" slicer-src \
    && git -C slicer-src fetch --depth 1 origin "${CURA_ENGINE_REF}" \
    && git -C slicer-src checkout --detach "${CURA_ENGINE_REF}"

ENV CURA_SKIP_SYSTEM_DEPS=true \
    PLATFORM=linux \
    ARCH=x86-64

RUN slicers/Cura/steps/install-deps.sh
RUN --mount=type=cache,target=/workspace/.cura-conan-home/p \
    slicers/Cura/steps/build-deps.sh
RUN --mount=type=cache,target=/workspace/.cura-conan-home/p \
    slicers/Cura/steps/build.sh
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN slicers/Cura/tests/test-artifacts.sh /workspace/cura-resources/resources
RUN --mount=type=cache,target=/workspace/.cura-conan-home/p \
    slicers/Cura/steps/package-binary.sh

FROM ubuntu:24.04 AS smoke
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=build /workspace/slicer-src/build/slicer_out /workspace/bundle
COPY slicers/Cura/tests /workspace/slicers/Cura/tests
COPY slicers/Cura/tools/generate-smoke-settings.py /workspace/slicers/Cura/tools/generate-smoke-settings.py
RUN /workspace/slicers/Cura/tests/smoke-bundle.sh /workspace/bundle
CMD ["/workspace/slicers/Cura/tests/smoke-bundle.sh", "/workspace/bundle"]

FROM scratch AS bundle
COPY --from=build /workspace/slicer-src/build/slicer_out /
