# tract7dt Documentation

`tract7dt` is a 7DT image-specific Tractor photometry pipeline.

This documentation is the primary reference for project behavior and configuration. The repository `README.md` is intentionally concise and links here for details.

## What This Pipeline Does

At a high level, `tract7dt run --config ...` executes:

1. Load and validate inputs (catalog, image list, FITS headers, WCS/shape checks).
2. Build ePSF products per ePSF grid cell and per band.
3. Build patch definitions.
4. Build patch input payloads.
5. Run Tractor patch subprocesses.
6. Merge patch outputs into a final catalog.

## Key Behavior Highlights

- **Path resolution**
  - Input relative paths resolve against the YAML config directory.
  - Output relative paths resolve against `outputs.work_dir`.

- **Catalog filtering before fitting**
  - Crop filtering removes out-of-region sources.
  - Optional saturation filtering removes near-saturated sources.
  - Exclusion reason flags are preserved in final merged output (`excluded_*` columns).

- **Type handling**
  - If `TYPE` is missing for fitting, the fallback is `patch_run.gal_model`.
  - Overlay diagnostics render unknown type sources as `UNKNOWN` with fallback context.

- **Performance controls**
  - Worker controls for load stage are configurable:
    - `performance.frame_prep_workers`
    - `performance.white_stack_workers`
  - ePSF/patch pruning:
    - `epsf.skip_empty_epsf_patches`
    - `patch_inputs.skip_empty_patch`

## Documentation Map

- **Installation**: environment and dependency setup
- **Quick Start**: basic run flow
- **Commands**: CLI commands and what each does
- **Configuration**: YAML model and parameter reference
- **Pipeline Behavior**: detailed runtime behavior
- **Performance Tuning**: practical speed knobs and trade-offs
- **Outputs**: generated files, key columns, diagnostics
- **Sample Data**: download command behavior and data terms
- **Troubleshooting**: common failure modes and checks

## Versioning and Source of Truth

Behavior described here is aligned with the current codebase under `tract7dt/`.

When behavior and docs drift, code is authoritative. This docs set is intended to minimize that drift by documenting implementation-level behavior explicitly.
