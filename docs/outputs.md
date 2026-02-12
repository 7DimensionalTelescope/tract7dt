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
│   │       ├── meta.json         # ePSF build metadata
│   │       ├── psfstars.png      # PSF star selection overlay
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
│           ├── src_000000.png              # Per-source montage
│           └── ...
├── cropped_images/           # Diagnostic plots
│   ├── white_before_crop.png
│   ├── white_after_crop.png
│   └── white_overlay.png
└── output_catalog.csv        # Final merged catalog
```

## Per-Patch Output Directory

Each patch produces a subdirectory under `outputs/` named by its patch tag (e.g. `r05_c02_pr04_pc00`). Contents:

| File | Description |
|------|-------------|
| `<tag>.log` | Tractor optimization log: iteration-by-iteration `dlnp` values, convergence status, PSF choices. |
| `<tag>.runner.log` | Full subprocess command line and captured stdout/stderr. Useful for debugging crashes. |
| `<tag>_cat_fit.csv` | Per-patch fitted catalog. Contains all input columns plus fit result columns. |
| `meta.json` | Patch metadata including PSF audit (which bands used ePSF vs fallback), optimizer summary (`niters`, `converged`, `hit_max_iters`), and patch geometry. |
| `patch_overview.png` | Three-panel plot: Data (white stack), Model+Sky, Residual. Source positions shown as crosshairs. |
| `cutouts/src_NNNNNN.png` | Per-source cutout montage (one row per panel: Data, Model+Sky, Residual; one column per band). Crosshairs show original (lime, dashed) and fitted (magenta, solid) positions. |

---

## Final Merged Catalog

**File:** `output_catalog.csv` (or as set by `outputs.final_catalog`)

The merge stage performs a **left join** from the original input catalog onto the concatenated patch fit results, using the merge key (`ID` if present, otherwise `RA`+`DEC`).

As a result:

- **Every row** in the original input catalog appears in the output, even if the source was excluded from fitting.
- Fit columns are **empty (`NaN`)** for sources that were excluded or not assigned to any patch.

### Input Catalog Columns (preserved)

All columns from the original input catalog are preserved in their original form. These include `ID`, `RA`, `DEC`, `TYPE`, and any `FLUX_*`, `ELL`, `THETA`, `Re` columns you provided.

### Exclusion Tracking Columns

These columns are added by the pipeline to indicate why a source was excluded from fitting.

| Column | Type | Description |
|--------|------|-------------|
| `excluded_crop` | bool | `True` if the source was removed by crop filtering (fell outside the crop margin). |
| `excluded_saturation` | bool | `True` if the source was removed by saturation-cut filtering (near saturated pixels). |
| `excluded_any` | bool | `True` if excluded by **any** reason (logical OR of all exclusion flags). |
| `excluded_reason` | string | Human-readable exclusion reason. Values: `"crop"`, `"saturation"`, `"crop+saturation"`, or `""` (not excluded). |

### Fitted Position Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `x_pix_patch_fit` | float | pixels | Fitted x position in the local patch coordinate frame. |
| `y_pix_patch_fit` | float | pixels | Fitted y position in the local patch coordinate frame. |
| `x_pix_white_fit` | float | pixels | Fitted x position in the full (post-crop) white-stack coordinate frame. Computed as `x_pix_patch_fit + x0_roi`. |
| `y_pix_white_fit` | float | pixels | Fitted y position in the full (post-crop) white-stack coordinate frame. Computed as `y_pix_patch_fit + y0_roi`. |
| `RA_fit` | float | degrees (ICRS) | Fitted right ascension, computed from `(x_pix_white_fit, y_pix_white_fit)` via WCS. |
| `DEC_fit` | float | degrees (ICRS) | Fitted declination, computed from `(x_pix_white_fit, y_pix_white_fit)` via WCS. |

### Fitted Flux Columns

For each band `{band}` present in the images (matching `FILTER` header values):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `FLUX_{band}_fit` | float | scaled counts (ZP = `zp_ref`) | Fitted flux in the given band. In the reference zeropoint system: `mag = -2.5 * log10(FLUX) + zp_ref`. |
| `FLUXERR_{band}_fit` | float | scaled counts (ZP = `zp_ref`) | Flux uncertainty (1-sigma), derived from the Tractor's parameter variance at the optimized solution. |

!!! note "Flux system"
    All fitted fluxes are in the scaled system defined by `image_scaling.zp_ref` (default 25.0). To convert to AB magnitudes: `m_AB = -2.5 * log10(FLUX_{band}_fit) + 25.0`.

### Fitted Morphology Columns

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

!!! note "Morphology columns for point sources"
    For `stype_fit = "star"`, all morphology columns (`sersic_n_fit`, `re_pix_fit`, `ab_fit`, `phi_deg_fit`, `ELL_fit`, `Re_fit`, `THETA_fit`) are `NaN`.

### Optimizer Diagnostic Columns

| Column | Type | Description |
|--------|------|-------------|
| `opt_converged` | bool | `True` if the optimizer converged (dlnp dropped below `patch_run.dlnp_stop`). |
| `opt_hit_max_iters` | bool | `True` if the optimizer did NOT converge and effectively exhausted iteration budget (iterations ≥ `n_opt_iters - flag_maxiter_margin`). |
| `opt_niters` | int | Number of optimizer iterations actually performed. |
| `opt_last_dlnp` | float | Last delta-log-probability value from the optimizer. |

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

### Source Cutout Montages

`cutouts/src_NNNNNN.png` shows a grid of panels: one column per band, three rows (Data, Model+Sky, Residual). Crosshairs mark:

- **Magenta solid** — fitted position
- **Lime dashed** — original (input) position
- **Deepskyblue X** — other fitted sources in the cutout
- **Orange X** — other original source positions in the cutout

---

## Merge Behavior Details

The merge stage:

1. Reads the original input catalog.
2. Attaches exclusion flags (crop, saturation) using the merge key.
3. Collects all `*_cat_fit.csv` files from per-patch output directories.
4. Concatenates fit results and checks for duplicate merge keys (raises `ValueError` if found).
5. Performs a **left join**: `base_catalog.merge(fit_results, how="left", on=key_cols)`.
6. Optionally fills `RA_fit`/`DEC_fit` using WCS from `merge.wcs_fits` or the first loaded image.

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
