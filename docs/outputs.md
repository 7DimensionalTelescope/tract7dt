# Outputs

This page documents every output artifact produced by the pipeline, with particular emphasis on the final merged catalog columns and their physical units.

## Directory Layout

With default settings, `outputs.work_dir` contains:

```
work_dir/
├── EPSFs/                    # ePSF artifacts
│   ├── <band>/
│   │   └── r00_c00/
│   │       ├── epsf.npy          # ePSF array (normalized)
│   │       ├── epsf.png          # ePSF stamp visualization
│   │       ├── epsf_growth_curve.png      # Encircled-energy diagnostic
│   │       ├── epsf_residual_diagnostics.png  # Star minus ePSF residual diagnostics
│   │       ├── meta.json         # ePSF build metadata
│   │       ├── psfstars.png      # PSF star selection overlay
│   │       ├── psfstars_bgsub.png  # PSF star overlay on background-subtracted patch
│   │       ├── background_diagnostics.png  # Raw/background/bg-sub patch panels
│   │       ├── star_local_background_diagnostics.png  # Per-star raw/annulus/bg-sub diagnostics
│   │       ├── extracted_stars.png   # Extracted star cutout montage
│   │       └── used_stars.png        # Fitted star cutout montage
│   └── epsf_summary.json    # Summary across all bands and cells
├── patches/                  # Patch geometry definitions
│   ├── patches.csv
│   └── patches.json
├── patch_payloads/           # Compressed per-patch input data
│   └── r00_c00_pr00_pc00.pkl.gz
├── outputs/                  # Per-patch Tractor fit results
│   └── r00_c00_pr00_pc00/
│       ├── r00_c00_pr00_pc00.log          # Patch fit log
│       ├── r00_c00_pr00_pc00.runner.log   # Subprocess stdout/stderr
│       ├── r00_c00_pr00_pc00_cat_fit.csv  # Patch fit catalog
│       ├── meta.json                       # Patch metadata + PSF audit
│       ├── patch_overview.png              # Data/Model/Residual overview
│       └── cutouts/
│           ├── src_{ID}.png                # Per-source montage (named by source ID)
│           └── ...
├── cropped_images/           # Diagnostic plots
│   ├── white_before_crop.png
│   ├── white_after_crop.png
│   └── white_overlay.png
├── ZP/                       # Zero-point calibration (if zp.enabled)
│   ├── input_catalog_*_with_Gaia.csv  # Augmented input catalog
│   ├── gaia_augmentation_overlay.png  # Source selection overlay
│   ├── zp_vs_mag__<band>.png         # Per-band ZP diagnostic plots
│   ├── zp_summary.csv                # Per-band ZP summary
│   └── zp_all_bands__per_object.csv  # Per-star ZP values
├── config_used_{timestamp}.yaml      # Snapshot of the config used for this run
├── {command}.log                     # Per-command log file (e.g. run.log, merge.log)
└── output_catalog.csv        # Final merged catalog
```

## ePSF Patch Diagnostics

Each `EPSFs/<band>/<epsf_tag>/` directory contains ePSF QA artifacts:

| File | Description |
|------|-------------|
| `psfstars.png` | PSF-star selection overlay on the raw ePSF cell image. |
| `psfstars_bgsub.png` | PSF-star selection overlay on the background-subtracted ePSF cell image. |
| `background_diagnostics.png` | Three panels: raw patch, estimated background map, raw-minus-background. Written only when `epsf.save_patch_background_diagnostics: true`. |
| `star_local_background_diagnostics.png` | Per-star diagnostics (one row per used star): raw stamp, annulus sample pixels, background-subtracted stamp. Written only when `epsf.save_star_local_background_diagnostics: true` and local-background subtraction is enabled. If `id_label` is missing/duplicated, this file is skipped with a warning. |
| `epsf_growth_curve.png` | Encircled-energy growth curve of the final ePSF stamp. |
| `epsf_residual_diagnostics.png` | Median star/model/residual panel plus normalized residual histogram. |

Per-cell `meta.json` includes `local_bkg_diagnostics.local_bkg_diag_status` to indicate whether `star_local_background_diagnostics.png` was written (`ok`) or skipped (`skipped_*`).

## Per-Patch Output Directory

Each patch produces a subdirectory under `outputs/` named by its patch tag (e.g. `r05_c02_pr04_pc00`). Contents:

| File | Description |
|------|-------------|
| `<tag>.log` | Tractor optimization log: iteration-by-iteration `dlnp` values, convergence status, PSF choices. |
| `<tag>.runner.log` | Full subprocess command line and captured stdout/stderr. Useful for debugging crashes. |
| `<tag>_cat_fit.csv` | Per-patch fitted catalog. Contains all input columns plus fit result columns. |
| `meta.json` | Patch metadata including PSF audit (which bands used ePSF vs fallback), patch geometry, and fitting mode. In multi-band mode, also includes optimizer summary (`niters`, `converged`, `hit_max_iters`); in single-band mode, per-band optimizer diagnostics are in the CSV only. |
| `patch_overview.png` | Three-panel plot: Data (white stack), Model+Sky, Residual. Source positions shown as crosshairs. |
| `cutouts/src_{ID}.png` | Per-source cutout montage (one row per panel: Data, Model+Sky, Residual; one column per band). Crosshairs show original (lime, dashed) and fitted (magenta, solid) positions. Filename uses the source `ID` (e.g. `src_00019.png`, `src_gaia_12345.png`). If `ID` is absent, falls back to zero-padded index (`src_000000.png`). Filesystem-unsafe characters in the ID are replaced with underscores. |

---

## Final Merged Catalog

**File:** `output_catalog.csv` (or as set by `outputs.final_catalog`)

The merge stage performs a **left join** from the base catalog onto the concatenated patch fit results, using the merge key (`ID` if present, otherwise `RA`+`DEC`). The base catalog is the augmented catalog (when `zp.enabled`) or the original input catalog (otherwise). Either way, it includes all sources — both active and excluded — with their flag columns.

As a result:

- **Every row** in the base catalog appears in the output, even if the source was excluded from fitting.
- Fit columns are **empty (`NaN`)** for sources that were excluded or not assigned to any patch.

### Input Catalog Columns (preserved)

All columns from the base catalog are preserved in their original form. These include `ID`, `RA`, `DEC`, `TYPE`, and any `FLUX_*`, `ELL`, `THETA`, `Re` columns you provided, plus `gaia_source_id` and exclusion flag columns added during the load and augmentation stages.

### Exclusion Tracking Columns

These columns are added by the pipeline to indicate why a source was excluded from fitting.

| Column | Type | Description |
|--------|------|-------------|
| `excluded_crop` | bool | `True` if the source was **flagged** by crop filtering (fell outside the crop margin). |
| `excluded_saturation` | bool | `True` if the source was **flagged** by saturation-cut filtering (near saturated pixels). |
| `excluded_any` | bool | `True` if excluded by **any** reason (logical OR of all exclusion flags). |
| `excluded_reason` | string | Human-readable exclusion reason. Values: `"crop"`, `"saturation"`, `"crop+saturation"`, or `""` (not excluded). |

### Fitted Position Columns

The column names depend on the fitting mode (`enable_multi_band_simultaneous_fitting`).

**Multi-band simultaneous fitting** (default, `true`): position is shared across bands.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `x_pix_patch_fit` | float | pixels | Fitted x position in the local patch coordinate frame. |
| `y_pix_patch_fit` | float | pixels | Fitted y position in the local patch coordinate frame. |
| `x_pix_white_fit` | float | pixels | Fitted x position in the full (post-crop) white-stack coordinate frame. Computed as `x_pix_patch_fit + x0_roi`. |
| `y_pix_white_fit` | float | pixels | Fitted y position in the full (post-crop) white-stack coordinate frame. Computed as `y_pix_patch_fit + y0_roi`. |
| `RA_fit` | float | degrees (ICRS) | Fitted right ascension, computed from `(x_pix_white_fit, y_pix_white_fit)` via WCS. |
| `DEC_fit` | float | degrees (ICRS) | Fitted declination, computed from `(x_pix_white_fit, y_pix_white_fit)` via WCS. |

**Single-band independent fitting** (`false`): position is fitted independently per band. For each band `{band}`:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `x_pix_patch_{band}_fit` | float | pixels | Fitted x position in the local patch frame for this band. |
| `y_pix_patch_{band}_fit` | float | pixels | Fitted y position in the local patch frame for this band. |
| `x_pix_white_{band}_fit` | float | pixels | Fitted x position in the white-stack frame for this band. |
| `y_pix_white_{band}_fit` | float | pixels | Fitted y position in the white-stack frame for this band. |
| `RA_{band}_fit` | float | degrees (ICRS) | Fitted right ascension for this band. |
| `DEC_{band}_fit` | float | degrees (ICRS) | Fitted declination for this band. |

### Fitted Flux Columns

For each band `{band}` present in the images (matching `FILTER` header values). Column names are the same in both fitting modes:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `FLUX_{band}_fit` | float | scaled counts (nominal ZP = `zp_ref`) | Fitted flux in the given band. Images are scaled so that the nominal ZP equals `zp_ref`, but see note below. |
| `FLUXERR_{band}_fit` | float | scaled counts (nominal ZP = `zp_ref`) | Flux uncertainty (1-sigma), derived from the Tractor's parameter variance at the optimized solution. |

!!! note "Flux system and approximate vs calibrated magnitudes"
    All fitted fluxes are in the scaled system defined by `image_scaling.zp_ref` (default 25.0). The formula `m_approx = -2.5 * log10(FLUX_{band}_fit) + zp_ref` gives only an **approximate** AB magnitude. This approximation assumes `ZP_AUTO` in the FITS header is the exact zero-point, but in practice `ZP_AUTO` has errors, and the Tractor photometry method differs from whatever method was used to derive `ZP_AUTO`. For **calibrated** AB magnitudes, use the `MAG_{band}_fit` column (when `zp.enabled`), which applies the ZP derived from Gaia-matched stars: `MAG_{band}_fit = ZP_median - 2.5 * log10(FLUX_{band}_fit)`. The offset `ZP_median - zp_ref` reflects primarily the error in `ZP_AUTO`. When ZP calibration is well-behaved, this offset is small. Note that `ZP_median` and `ZP_AUTO` live in different flux systems (scaled vs original), so comparing them directly is not meaningful — see [Pipeline Behavior — ZP Calibration](pipeline-behavior.md#why-zp-calibration-is-needed).

### Fitted Morphology Columns

**Multi-band simultaneous fitting** (default, `true`): morphology is shared across bands.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `stype_fit` | string | — | Fitted source type: `"star"`, `"exp"`, `"dev"`, or `"sersic"`. |
| `sersic_n_fit` | float | dimensionless | Fitted Sersic index (only for `sersic` type; `NaN` otherwise). |
| `re_pix_fit` | float | pixels | Fitted effective (half-light) radius in the Tractor's internal parameterization. |
| `ab_fit` | float | dimensionless | Fitted axis ratio `b/a` (range 0–1). |
| `phi_deg_fit` | float | degrees | Fitted position angle in the Tractor's internal convention (modulo 180). |
| `ELL_fit` | float | dimensionless | Fitted ellipticity: `1 - ab_fit`. Comparable to the input `ELL` column. |
| `Re_fit` | float | pixels | Fitted effective radius (same as `re_pix_fit`; provided for naming symmetry with input `Re`). |
| `THETA_fit` | float | degrees | Fitted position angle, approximately converted back to SExtractor convention: `phi_deg_fit - 90`. Comparable to the input `THETA` column. |

**Single-band independent fitting** (`false`): morphology is fitted independently per band. For each band `{band}`:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `stype_fit` | string | — | Source type (same across bands, from input `TYPE` column). |
| `sersic_n_{band}_fit` | float | dimensionless | Fitted Sersic index for this band. |
| `re_pix_{band}_fit` | float | pixels | Fitted effective radius for this band. |
| `ab_{band}_fit` | float | dimensionless | Fitted axis ratio for this band. |
| `phi_deg_{band}_fit` | float | degrees | Fitted position angle for this band. |
| `ELL_{band}_fit` | float | dimensionless | Fitted ellipticity for this band: `1 - ab_{band}_fit`. |
| `Re_{band}_fit` | float | pixels | Fitted effective radius for this band. |
| `THETA_{band}_fit` | float | degrees | Fitted position angle for this band (SExtractor convention). |

!!! note "Morphology columns for point sources"
    For `stype_fit = "star"`, all morphology columns are `NaN` (in both fitting modes).

### Optimizer Diagnostic Columns

**Multi-band simultaneous fitting** (default, `true`): one set of optimizer diagnostics per source.

| Column | Type | Description |
|--------|------|-------------|
| `opt_converged` | bool | `True` if the optimizer converged (dlnp dropped below `patch_run.dlnp_stop`). |
| `opt_hit_max_iters` | bool | `True` if the optimizer did NOT converge and effectively exhausted iteration budget (iterations ≥ `n_opt_iters - flag_maxiter_margin`). |
| `opt_niters` | int | Number of optimizer iterations actually performed. |
| `opt_last_dlnp` | float | Last delta-log-probability value from the optimizer. |

**Single-band independent fitting** (`false`): per-band optimizer diagnostics. For each band `{band}`:

| Column | Type | Description |
|--------|------|-------------|
| `opt_converged_{band}` | bool | Whether the optimizer converged for this band. |
| `opt_hit_max_iters_{band}` | bool | Whether the optimizer exhausted iterations for this band. |
| `opt_niters_{band}` | int | Number of iterations for this band. |
| `opt_last_dlnp_{band}` | float | Last dlnp for this band. |

### Patch and PSF Audit Columns

| Column | Type | Description |
|--------|------|-------------|
| `patch_tag` | string | Patch identifier (e.g. `r05_c02_pr04_pc00`). |
| `epsf_tag` | string | Parent ePSF cell identifier (e.g. `r05_c02`). |
| `psf_min_epsf_nstars_for_use` | int | Quality gate threshold used for this patch. |
| `psf_used_epsf_band_count` | int | Number of bands that used a real ePSF. |
| `psf_fallback_band_count` | int | Number of bands that used a fallback PSF. |
| `psf_low_star_band_count` | int | Number of bands where ePSF existed but was rejected due to low star count. |
| `psf_fallback_bands` | string | Comma-separated list of bands that used fallback PSF. |
| `psf_low_star_bands` | string | Comma-separated list of bands with low-star ePSF rejection. |
| `psf_frozen_bands_for_optimizer` | string | Comma-separated list of bands where GaussianMixturePSF params were frozen for optimizer stability. |
| `psf_fallback_reasons_json` | string (JSON) | JSON dict mapping band → fallback reason string. |

### Internal Coordinate Columns

These columns are added during patch input building and carried through to the output. They are useful for debugging spatial assignments.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `x_pix_white` | float | pixels | Source x position in the white-stack frame (from WCS projection of input RA/DEC). |
| `y_pix_white` | float | pixels | Source y position in the white-stack frame. |
| `x_pix_patch` | float | pixels | Source x position in the local patch frame (= `x_pix_white - x0_roi`). |
| `y_pix_patch` | float | pixels | Source y position in the local patch frame. |

### Gaia Augmentation Column

When `zp.enabled: true`, the augmented catalog adds this column:

| Column | Type | Description |
|--------|------|-------------|
| `gaia_source_id` | string | Gaia DR3 `source_id` for sources matched to (or injected from) the GaiaXP synphot catalog. Empty string for non-Gaia sources. |

### Zero-Point Calibrated Magnitude Columns

When `zp.enabled: true`, the ZP computation stage adds these columns for each band `{band}`:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `MAG_{band}_fit` | float | mag (AB) | AB magnitude: `ZP_median - 2.5 * log10(FLUX_{band}_fit)`. `NaN` for sources with non-positive flux. |
| `MAGERR_{band}_fit` | float | mag | Magnitude error: `sqrt((2.5/ln10 * FLUXERR/FLUX)^2 + ZP_err^2)`. Propagates both flux error and ZP uncertainty. `NaN` if flux error is missing. |

---

## Diagnostic Plots

### White-Stack Diagnostics

| File | Description |
|------|-------------|
| `cropped_images/white_before_crop.png` | White-stack image with crop box overlaid (green rectangle). Generated before crop is applied. |
| `cropped_images/white_after_crop.png` | White-stack image after crop. |
| `cropped_images/white_overlay.png` | Post-crop white stack with source positions color-coded by TYPE: cyan=STAR, magenta=GAL/EXP/DEV/SERSIC, yellow squares=UNKNOWN, red X=saturation-excluded, gray=crop-excluded (legend only). |

### Patch Overview

`patch_overview.png` in each patch output directory shows a three-panel view (Data, Model+Sky, Residual) of the white-combined patch images. Source positions are marked:

- **Orange X** — original (input catalog) positions
- **Deepskyblue X** — fitted positions
- **Cyan line** — connecting line from each original position to its fitted position(s)

In single-band mode, each source has **multiple blue markers** (one per band, with reduced alpha/linewidth to avoid clutter), each connected to the single orange marker by a cyan line. In multi-band mode, there is one blue marker per source.

### Source Cutout Montages

`cutouts/src_{ID}.png` shows a grid of panels: one column per band, three rows (Data, Model+Sky, Residual). The filename uses the source's `ID` value (e.g. `src_00019.png`, `src_gaia_2518065009826205440.png`). If `ID` is absent, a zero-padded index is used as fallback. Markers for the **center source**:

- **Magenta solid crosshair** — fitted position
- **Lime dashed crosshair** — original (input) position

Markers for **other sources** in the cutout:

- **Deepskyblue X** — fitted position
- **Orange X** — original position
- **Cyan line** — connecting line from each original to fitted position

In single-band mode, each band column shows that band's own independently fitted position for the crosshairs and markers.

### ZP Diagnostics

When `zp.enabled: true`, the `ZP/` directory contains:

| File | Description |
|------|-------------|
| `input_catalog_*_with_Gaia.csv` | Augmented input catalog with Gaia-matched `gaia_source_id` column and (if `inject_gaia_sources: true`) injected Gaia source rows. |
| `gaia_augmentation_overlay.png` | White-stack overlay showing original sources (cyan circles), Gaia-matched originals (lime squares), injected Gaia sources (magenta circles, only when `inject_gaia_sources: true`), saturation-excluded Gaia sources (red X), saturation-excluded input sources (red X), crop-excluded input sources (orange X), and the selection bounding box (yellow dashed). |
| `zp_vs_mag__{band}.png` | Per-band ZP diagnostic scatter plot: ZP vs GaiaXP synphot AB magnitude. Shows non-converged (gray diamonds), sigma-clipped (red X), kept (black circles with error bars), median line, and MAD band. |
| `zp_summary.csv` | Per-band summary: `n_raw`, `n_kept`, `n_clipped`, `n_nonconverged`, `clip_sigma`, `zp_median`, `zp_err_mad`. |
| `zp_all_bands__per_object.csv` | Per-star ZP values for all bands, with `is_kept`/`is_clipped` flags. |

---

## Merge Behavior Details

The merge stage:

1. Reads the base catalog (augmented catalog if ZP is enabled, otherwise the original input catalog). Exclusion flag columns (`excluded_crop`, `excluded_saturation`) are already present from the load stage and are used directly.
2. Computes `excluded_any` and `excluded_reason` from the flag columns.
3. Collects all `*_cat_fit.csv` files from per-patch output directories.
4. Concatenates fit results and handles duplicate merge keys (drops duplicates with a warning; rare boundary sources may appear in two patches).
5. Performs a **left join**: `base_catalog.merge(fit_results, how="left", on=key_cols)`.
6. Optionally computes sky coordinates from fitted pixel positions using WCS from `merge.wcs_fits` or the first loaded image. In multi-band mode: `RA_fit`/`DEC_fit`. In single-band mode: `RA_{band}_fit`/`DEC_{band}_fit` for each band.

### Merge Key Selection

- If `ID` column exists → use `ID` as the sole merge key.
- Otherwise → use `(RA, DEC)` as a composite merge key.

### What "merge-missing" Means

The merge log reports `merge-missing (left join rows without fit): N`. This count includes:

- Sources excluded by crop filtering.
- Sources excluded by saturation filtering.
- Sources with `NaN` coordinates that could not be assigned to patches.
- Sources in patches that failed or crashed.

Check the `excluded_*` columns to distinguish intentional exclusions from unexpected failures.

## Merge Logging

The merge stage logs:

- Output file path
- Row and column count
- Exclusion counts (if exclusion columns are present): crop, saturation, any
- Merge-missing count (left-join rows without fit results)
