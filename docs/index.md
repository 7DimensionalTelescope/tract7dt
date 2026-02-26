# Tract7DT Documentation

`tract7dt` is a 7DT image-specific Tractor photometry pipeline.

This documentation is the primary reference for project behavior and configuration. The repository `README.md` is intentionally concise and links here for details.

Hosted site: https://seoldh99.github.io/tract7dt/

## What This Pipeline Does

`tract7dt` performs forced multi-band photometry on 7-Dimensional Telescope (7DT) images using [the Tractor](https://github.com/dstndstn/tractor) — a probabilistic astronomical source modeling framework. Given a catalog of source positions and a set of aligned multi-band images, it simultaneously fits source models (point sources and galaxies) to all bands, producing optimized flux measurements, morphological parameters, and associated uncertainties.

At a high level, `tract7dt run --config ...` executes:

1. **Load and validate inputs** — read the source catalog, load FITS images, build the white-stack coadd, flag sources affected by crop margins or saturation.
2. **Augment catalog with Gaia sources** *(if `zp.enabled`)* — match input sources against GaiaXP synphot catalog, backfill missing fluxes, and optionally inject unmatched Gaia sources as additional rows for zero-point calibration.
3. **Build ePSF products** — construct an empirical PSF per band in each spatial grid cell, using SEP detections and optionally Gaia DR3 XP synthetic photometry.
4. **Build patch definitions** — subdivide the image into spatial patches for parallel processing.
5. **Build patch input payloads** — assign non-excluded sources to patches and serialize per-patch data bundles.
6. **Run Tractor patch subprocesses** — fit source models independently in each patch.
7. **Merge patch outputs** — combine all patch fit results into a single final catalog, preserving excluded sources with flag columns.
8. **Compute zero-point calibration** *(if `zp.enabled`)* — derive per-band ZP from Gaia-matched stars and apply AB magnitudes to all sources.

## Key Behavior Highlights

- **Path resolution**
  - Input relative paths resolve against the YAML config directory.
  - Output relative paths resolve against `outputs.work_dir`.

- **Input catalog format**
  - CSV format with required `RA`/`DEC` columns (degrees, ICRS).
  - Optional `ID`, `TYPE`, `FLUX_{band}`, `ELL`, `THETA`, `Re` columns.
  - See [Input Catalog Reference](input-catalog.md) for comprehensive format documentation.

- **Catalog filtering before fitting**
  - Crop filtering flags out-of-region sources (`excluded_crop`).
  - Optional saturation filtering flags near-saturated sources (`excluded_saturation`).
  - Flagged sources remain in the catalog and final output with exclusion reason columns (`excluded_*`).

- **Source model selection**
  - `TYPE` column maps to Tractor models: `STAR` → PointSource, `EXP` → ExpGalaxy, `DEV` → DevGalaxy, `SERSIC` → SersicGalaxy.
  - Missing/unknown types fall back to `patch_run.gal_model` (default: `exp`).

- **PSF handling**
  - Empirical PSF (ePSF) is built per band per spatial cell from real stars.
  - Quality gate: ePSFs with too few stars fall back to analytic models (Moffat, Gaussian mixture).
  - PSF audit information is recorded per-source in the output catalog.

- **Performance controls**
  - Worker controls for load stage: `performance.frame_prep_workers`, `performance.white_stack_workers`.
  - ePSF/patch pruning: `epsf.skip_empty_epsf_patches`, `patch_inputs.skip_empty_patch`.
  - Fitting parallelism: `patch_run.max_workers`, `patch_run.threads_per_process`.
  - Diagnostic plot toggles: `epsf.save_star_local_background_diagnostics` (major impact — one plot per star per cell), `epsf.save_patch_background_diagnostics`, `epsf.save_growth_curve`, `epsf.save_residual_diagnostics`, `patch_run.no_cutouts`, `patch_run.no_patch_overview`. Disabling these can significantly reduce I/O and runtime.

## Documentation Map

| Page | Description |
|------|-------------|
| [Installation](installation.md) | Environment and dependency setup |
| [Quick Start](quickstart.md) | Step-by-step first run guide |
| [Commands](commands.md) | CLI commands, stage dependencies, iteration patterns |
| [Input Catalog Reference](input-catalog.md) | **Comprehensive guide to input files: catalog format, column units, FITS requirements, GaiaXP catalog, validation rules** |
| [Configuration](configuration.md) | Complete YAML parameter reference with types, units, and defaults |
| [Pipeline Behavior](pipeline-behavior.md) | Detailed runtime behavior of each stage |
| [Performance Tuning](performance.md) | Practical speed knobs and trade-offs |
| [Outputs](outputs.md) | Output directory layout, catalog columns, diagnostic plots |
| [Sample Data](sample-data.md) | Download command behavior and data terms |
| [Troubleshooting](troubleshooting.md) | Common errors, warnings, and fixes |

## Versioning and Source of Truth

Behavior described here is aligned with the current codebase under `tract7dt/`.

When behavior and docs drift, code is authoritative. This docs set is intended to minimize that drift by documenting implementation-level behavior explicitly.
