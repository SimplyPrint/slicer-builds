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

## Licensing

This repository — the build scripts, CI workflows, shared tools, and patches — is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0), with no additional terms.

The patches under `slicers/<Slicer>/patches/` modify upstream slicer source code and are therefore also governed by the corresponding upstream license. A copy of each upstream project's license is kept at `slicers/<Slicer>/LICENSE`:

| Slicer | Upstream repository | License |
| --- | --- | --- |
| AnycubicSlicerNext | [ANYCUBIC-3D/AnycubicSlicerNext](https://github.com/ANYCUBIC-3D/AnycubicSlicerNext) | [AGPL-3.0](slicers/AnycubicSlicerNext/LICENSE) |
| BambuStudio | [bambulab/BambuStudio](https://github.com/bambulab/BambuStudio) | [AGPL-3.0](slicers/BambuStudio/LICENSE) |
| CrealityPrint | [CrealityOfficial/CrealityPrint](https://github.com/CrealityOfficial/CrealityPrint) | [AGPL-3.0](slicers/CrealityPrint/LICENSE) |
| Cura | [Ultimaker/CuraEngine](https://github.com/Ultimaker/CuraEngine) + [Ultimaker/Cura](https://github.com/Ultimaker/Cura) | [AGPL-3.0](slicers/Cura/LICENSE.CuraEngine) + [LGPL-3.0](slicers/Cura/LICENSE.Cura) |
| ElegooSlicer | [ELEGOO-3D/ElegooSlicer](https://github.com/ELEGOO-3D/ElegooSlicer) | [AGPL-3.0](slicers/ElegooSlicer/LICENSE) |
| OrcaSlicer | [OrcaSlicer/OrcaSlicer](https://github.com/OrcaSlicer/OrcaSlicer) | [AGPL-3.0](slicers/OrcaSlicer/LICENSE) |
| PrusaSlicer | [prusa3d/PrusaSlicer](https://github.com/prusa3d/PrusaSlicer) | [AGPL-3.0](slicers/PrusaSlicer/LICENSE) |
| QIDIStudio | [QIDITECH/QIDIStudio](https://github.com/QIDITECH/QIDIStudio) | [AGPL-3.0](slicers/QIDIStudio/LICENSE) |
| SuperSlicer | [supermerill/SuperSlicer](https://github.com/supermerill/SuperSlicer) | [AGPL-3.0](slicers/SuperSlicer/LICENSE) |

Binaries built from the patched sources are covered works of the corresponding upstream slicer and remain AGPL-3.0. Generated configuration artifacts under `slicers/<Slicer>/out/` inherit the status of the slicer data they are derived from.
