# Slicer builds

Custom SimplyPrint slicer builds and generated slicer configuration artifacts.

## Repository layout

- `tools/` contains shared helper tools, including `apply_versioned_patches.sh` and `generate_defs_visibility`.
- `patches/` contains named patch sets shared by compatible forks. Manifests opt
  into these sets; a fix does not need to be copied into every slicer tree.
- `slicers/<Slicer>/steps/` contains build, run, dependency, and packaging scripts for each slicer.
- `slicers/<Slicer>/patches/` contains dump patches plus optional versioned
  patches. Patches needed by both build modes live under `patches/common/`;
  binary-only patches live under `patches/binary/`.
- `slicers/<Slicer>/out/` contains generated artifacts:
  - `.cache/` for workflow cache placeholders.
  - `nightly/` for nightly dumped configs.
  - `<version>/` for release dumped configs.
  - `_index.json` as the single version/build/config index for that slicer.

All slicer step scripts are run from the repository root with `slicer-src` and `slicer-out` available.

## Local Docker workflow

`slicerctl` is the supported local entry point. It discovers slicers from
`slicers/*/slicer.toml`, so local build and config matrices do not need
another hard-coded slicer list.

Build the reusable Ubuntu 24.04 builder, then build one slicer or the complete
native-architecture matrix:

```bash
./slicerctl image
./slicerctl list
./slicerctl build OrcaSlicer
./slicerctl build-all
./slicerctl build-all --workers 2
./slicerctl build-all --all-refs --workers 2
```

Binary fixes are applied by default. Use `--patches none` to compare clean
upstream behavior, or `--patches dump` for a config-generator development
checkout. `docker` and `podman` are both supported; set
`SLICER_CONTAINER_CLI` or pass `--container-cli` to select one explicitly.

`--ref` accepts an arbitrary tag, branch, or commit. `--source /path/to/local/clone`
seeds the local mirror from committed work while keeping
the same isolated checkout and cache layout. Commit a temporary development
branch first: uncommitted files are intentionally not copied into managed
checkouts.

Build results live below `.work/results/`, with managed checkouts separated by
upstream commit, architecture, patch-content hash, builder identity, and
toolchain settings. Re-running an identical build reuses the result. `--force`
reruns it while CMake, Conan, sccache, and dependency artifacts remain warm.
File locks make concurrent invocations safe, and identical dependency trees can
share a content-addressed cache across related forks. A changed dependency or
build recipe invalidates only the incompatible CMake tree; compiler, download,
Conan, and content-addressed dependency caches remain warm. Release manifests
also lock their default tag to its expected commit, while explicit `--ref`
development builds remain unrestricted. Metadata records include a digest of
the complete staged bundle and reject different bytes under the same build
identity. Each final bundle is moved into its fingerprint-owned artifact
directory, so switching between full-resource, pruned, stripped, or debug
packaging variants does not overwrite another reusable result. Corrupt cache
metadata or mutated artifacts are moved under `.work/quarantine/` before a
clean rebuild; genuinely different new bytes for an otherwise valid immutable
identity still fail as a reproducibility error.

Manifests may also declare commit-locked `[[supported_refs]]`. `matrix
--all-refs` and `build-all --all-refs` expand the
default plus every retained base, while a one-off arbitrary `--ref` remains
available for branch development. This replaces manually copying patches
between long-lived working trees.

When adjacent upstream versions accept the identical binary overlay, a
manifest can map the new ref to one canonical patch directory with
`[patches.binary_ref_aliases]`. Common patches still use the real ref. For
example, Orca `v2.4.1` reuses its verified `v2.4.2` thumbnail overlay without a
copied patch, while `v2.4.0` retains its larger version-specific stack. Patches
that apply across the whole maintained range belong in `patches/binary/all/`;
Bambu's no-`wxApp` thumbnail fix is shared this way.

Check every declared binary and config-dump patch stack without compiling or
creating working-tree checkouts:

```bash
./slicerctl verify-patches
./slicerctl verify-patches --slicer OrcaSlicer --ref v2.4.2 --mode binary
./slicerctl verify-patches --slicer OrcaSlicer --source /path/to/local/clone
```

The check resolves each ref once, enforces its locked commit, and applies the
ordered patches to isolated temporary Git indexes. Failures from other stacks
are collected by default; use `--fail-fast` for a short local iteration or
`--json` for machine-readable results. The scheduled CI gate adds
`--include-head`, which also checks current `HEAD` for manifests with an
enabled nightly lane while retaining the literal `HEAD` nightly-patch
selection.

## Fast, small CLI builds

The defaults favor repeatable deployable binaries:

- Anycubic, Creality, Elegoo, Orca, and QIDI configure with the definitions and
  generator used by their upstream Linux wrappers, then build only the main
  slicer target. This avoids profile-validator and discarded distribution
  builds without patching upstream build scripts. Creality deliberately keeps
  its upstream `ORCA_TOOLS=ON` and Breakpad configuration; QIDI is configured
  as a non-public build directly.
- PrusaSlicer and SuperSlicer disable GUI, tests, sandboxes, desktop integration,
  and STEP support for binary builds. Config-dump jobs explicitly turn GUI back
  on because their dump patches use GUI tabs; the command
  `./slicerctl build PrusaSlicer --patches dump` selects and caches that GUI
  mode automatically.
- Cura disables Arcus, plugins, benchmarks, extensive warnings, and dependency
  tests.
- Ninja, PCH, exact targets, persistent compiler/Conan storage, and content
  hashes keep rebuilds incremental.
- The default compile job count respects both CPU and available/cgroup memory.
  `--workers` enables bounded slicer-level concurrency; when `--jobs` is
  omitted, the safe job budget is divided between concurrent builds.
- Build-result metadata records monotonic dependency, compile, and packaging
  phase durations. A dependency-cache hit is reported separately and leaves
  that phase absent instead of presenting a misleading zero-second build.
- Date-stamped forks receive the upstream commit date before configuring, so
  `--force` cannot publish a different binary under the same cache identity.
- The shared direct-CMake helper keeps target-only mechanics consistent while
  each slicer's short build step records its fork-specific upstream configure
  contract. Changes to either layer invalidate the build cache. These bundles
  stage the upstream source resource tree directly instead of first making the
  distribution script's large temporary resource copy.
- Packaging strips ELF files and recursively copies only required private
  libraries from build/dependency roots. libc, GTK, WebKit, Mesa, and other
  libraries already supplied by the cloud runtime are not duplicated in every
  bundle. The staged ELF machine must match the requested architecture. Set
  `SLICER_STRIP=0` only when local debugging needs symbols.
- Every build result retains and cross-checks the staging inventory: private
  library count plus source, selected, omitted, and staged bytes for each
  top-level resource group.

The controller forwards `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `LDFLAGS`,
`CMAKE_GENERATOR`, `SLICER_PCH`, and `SLICER_STRIP` into Docker and includes
the applicable values in cache identity. The direct fork drivers also preserve
their upstream `ORCA_EXTRA_BUILD_ARGS`, `ELEGOO_EXTRA_BUILD_ARGS`, and
`ORCA_UPDATER_SIG_KEY` inputs with per-slicer scoping. For example:

```bash
SLICER_PCH=OFF CC=clang CXX=clang++ LDFLAGS=-fuse-ld=lld \
  ./slicerctl build OrcaSlicer --jobs 16
```

PCH-on is usually fastest for a clean or ordinary incremental build.
`SLICER_PCH=OFF` is supported by Anycubic, Elegoo, Orca, Prusa, and SuperSlicer
and can improve sccache reuse across frequently changing branches; it is
deliberately ignored for drivers that do not consume the setting.
Clang/lld can reduce link time, but should be measured per family. The repository
does not enable `-march=native`, full LTO, or fast-math globally: native tuning
breaks deployment portability, full LTO materially increases build time, and
fast-math can change geometric results. ThinLTO is worth testing only after the
real-profile slice matrix and output comparisons pass.

### Runtime resource pruning

Resource pruning is fail-safe and opt-in. By default every upstream resource is
copied. For an exact slicer/ref experiment, provide newline-delimited patterns;
the policy affects packaging identity but does not discard the warm compile
tree:

```bash
SLICER_RESOURCE_INCLUDES=$'flush/**\nfonts/**\ninfo/**\nprofiles/BBL/cli_config.json\nshaders/**' \
  ./slicerctl build OrcaSlicer --force
```

Unsafe, unmatched, or escaping patterns and symlinks fail before the previous
bundle is replaced. Packaging reports source, staged, and omitted byte counts,
including top-level resource groups with `tools/stage_bundle.py --json`.

Measurements against the current backend bundles show the available reduction
before archive compression:

| Slicer/version | Full resources | Conservative candidate | Reduction |
| --- | ---: | ---: | ---: |
| BambuStudio 02.07.01.62 | 386.36 MiB | 34.71 MiB | 91.02% |
| OrcaSlicer 2.4.2 | 213.19 MiB | 34.96 MiB | 83.60% |
| CrealityPrint 7.1.1 | 236.04 MiB | 57.26 MiB | 75.74% |
| ElegooSlicer 1.5.1.6 | 159.85 MiB | 35.26 MiB | 77.94% |
| PrusaSlicer 2.9.6 | 153.87 MiB | 0.09 MiB | 99.94% |

Those candidates completed real backend-adapter slices with the installed
binaries. They are not global defaults yet: the installed derived binaries have
the old thumbnail failure, and a single printer/profile does not prove dynamic
bed assets, fonts, multi-material resources, or future fork layouts. Keep new
slicers on copy-all, then enable an exact-ref policy only after production-shaped
STL/3MF, thumbnail, multi-material, and text/embossed tests pass.

## Adding and maintaining slicers

To onboard a slicer, add its license, `steps/` scripts, and one `slicer.toml`.
`./slicerctl list` validates required fields, supported architectures,
capabilities and step scripts. The entry then appears automatically in
`matrix` and `build-all`.

Keep source changes small and upstream-shaped:

1. Test the unpatched upstream ref with `--patches none`.
2. Put a dependency or build-driver fix needed by both modes in
   `patches/common/all/` or `patches/common/<ref>/`. Within each declared shared
   or local patch root, its common layer runs before its mode-specific layer;
   shared roots still run before the slicer-local overlay.
3. Put a binary-only fix that applies across versions in
   `patches/binary/all/` once.
4. Put a fix shared by multiple forks in a named root `patches/<set>/` and add
   that set to each manifest's `[patches].shared` list.
5. Use `patches/binary/<ref>/` only where upstream source really diverges;
   `HEAD` falls back to `nightly` when no explicit HEAD directory exists.
6. Run `slicerctl verify-patches` to check all retained bases without compiling.
   Use `slicerctl prepare` when an inspectable patched checkout is useful, then
   run the Docker build and inspect the staged bundle.
7. Submit behavior fixes upstream and remove local patches once every supported
   base includes the fix.

The same convention applies to dump patches. This keeps changes reviewable and
lets Orca-, Bambu-, and vendor-derived bases reuse the same workflow without
copying source trees or manually carrying patched working directories.

When retaining multiple upstream bases, add each one as a `[[supported_refs]]`
entry with its full commit ID. Keep only ref-specific patches that the expanded
`build-all --all-refs` matrix proves necessary.

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
