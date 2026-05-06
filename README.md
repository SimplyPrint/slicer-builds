# Slicer builds

Custom SimplyPrint slicer builds and generated slicer configuration artifacts.

## Repository layout

- `tools/` contains shared helper tools, including `apply_versioned_patches.sh` and `generate_defs_visibility`.
- `slicers/<Slicer>/steps/` contains build, run, dependency, and packaging scripts for each slicer.
- `slicers/<Slicer>/patches/` contains dump patches plus optional versioned patches. Binary-only patches live under `patches/binary/`.
- `slicers/<Slicer>/out/` contains generated artifacts:
  - `.cache/` for workflow cache placeholders.
  - `nightly/` for nightly dumped configs.
  - `<version>/` for release dumped configs.
  - `_index.json` as the single version/build/config index for that slicer.

All slicer step scripts are run from the repository root with `slicer-src` and `slicer-out` available.
