# Slicer builds

Custom builds of slicers for different purposes

## Dump slicer configs

Each slicer has steps configured in `dump_slicer_steps` all steps (scripts) are run from the root of the repository with
two folders available:

`slicer-src`

`slicer-out`

The `build.sh` and `run.sh` will use these folders respectively.

## Conditional visibility

RAG pipeline to construct "show-if" logic for printer defs based on slic3r `toggle_print_fff_options`.
