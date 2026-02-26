# Pipeline Behavior

This page describes the runtime behavior of each pipeline stage as implemented in code. Understanding this flow is essential for interpreting outputs and diagnosing issues.

## Pipeline Overview

The full pipeline (`tract7dt run`) executes up to eight sequential stages:

```
load_inputs → [augment_gaia] → build_epsf → build_patches → build_patch_inputs → run_patches → merge → [compute_zp]
```

Stages in brackets are conditional on `zp.enabled: true`.

Each stage can also be run independently via its own CLI command (see [Commands](commands.md)). Step commands that require state (`run-epsf`, `build-patches`, `build-patch-inputs`, `merge`) automatically run `load_inputs` and Gaia augmentation (when `zp.enabled`) to ensure the same catalog state as a full `run`.

### Config Snapshot

Before executing any pipeline or step command, the CLI saves a timestamped copy of the config file to `{work_dir}/config_used_{YYYYMMDD_HHMMSS}.yaml`. This provides a record of exactly which parameters were used for each run.

---

## Stage 1: Load Inputs and Validate

**Function:** `load_inputs()` in `pipeline.py`

This is the most complex stage. It loads all input data, validates consistency, builds derived products (white stack, masks), and applies pre-fit source filters.

### 1.1 Read Input Catalog

1. Read the CSV file at `inputs.input_catalog` using `pandas.read_csv()`.
2. Locate `RA` and `DEC` columns (case-insensitive lookup). Raise `ValueError` if not found.
3. Locate `TYPE` column (if present) for overlay diagnostics.
4. Locate `ID` column (if present) for merge key. If absent, use `(RA, DEC)` as composite key.
5. Initialize exclusion tracking flags (`excluded_crop`, `excluded_saturation`) for all sources.

### 1.2 Read Image List and Prepare Frames

1. Read image paths from `inputs.image_list_file` (one path per line, comments and blanks ignored).
2. For each FITS image, in parallel (using `performance.frame_prep_workers` threads):
   - Load the primary HDU data and header.
   - Validate required header keywords: `FILTER`, `ZP_AUTO`, `SKYSIG`, `EGAIN`.
   - Compute photometric scaling: `scale = 10^(-0.4 * (ZP_AUTO - zp_ref))`.
   - Apply scaling to image: `img_scaled = raw_image * scale`.
   - Build bad-pixel mask: non-finite pixels.
   - Build saturation mask: pixels where `raw_image >= SATURATE / 2` (if `SATURATE` is present).
   - Combine masks: `bad = ~finite | saturated`.
   - Build noise arrays:
     - Sky sigma: `sigma_sky_scaled = SKYSIG * scale` (inf where bad).
     - Source variance: `var_src = max(raw_image, 0) / EGAIN`.
     - Total sigma: `sigma_total_scaled = sqrt(SKYSIG^2 + var_src) * scale`.
   - Construct WCS from header.

### 1.3 Validate Shape and WCS Consistency

If `checks.require_same_shape: true`:

- All images must have identical `(NAXIS2, NAXIS1)` dimensions. Mismatch raises `RuntimeError`.

If `checks.require_wcs_alignment: true`:

- All images must have WCS that agrees with the first image within the configured tolerances (`checks.wcs_tolerance.*`).
- Checks: CRVAL, CRPIX, CD (or CDELT+PC), CTYPE.

### 1.4 Band Consistency Check

If the input catalog contains any `FLUX_*` columns:

- Extract band names from catalog columns (e.g. `FLUX_m400 → m400`).
- Extract band names from image `FILTER` headers.
- If these sets differ, raise `RuntimeError` with a detailed mismatch message.

### 1.5 Build White Stack

The white (inverse-variance-weighted coadd) stack is built in parallel row chunks:

```
white[y,x] = sum_bands(w * img) / sum_bands(w)
where w = 1 / sigma_sky^2 for good pixels, 0 for bad pixels
```

The white-noise map is: `sigma_white = sqrt(1 / sum_bands(w))`.

Parallelism is controlled by `performance.white_stack_workers`.

### 1.6 Apply Crop Filter (if enabled)

If `crop.enabled: true`:

1. Define crop box: `[margin, W-margin) x [margin, H-margin)` in pixels.
2. Project all source RA/DEC to pixel coordinates using the first image's WCS.
3. Sources outside the crop box are **flagged** with `excluded_crop = True`. They remain in the catalog but are excluded from downstream fitting.
4. Slice the white stack, per-band images, masks, sigma arrays, and WCS to the crop region.
5. Generate diagnostic plots (pre-crop, post-crop).

### 1.7 Apply Saturation Filter (if enabled)

If `source_saturation_cut.enabled: true`:

1. For each source, project RA/DEC to pixel coordinates.
2. Check a circular disk of radius `source_saturation_cut.radius_pix` pixels around each source.
3. If any pixel in the disk is flagged as saturated:
   - `require_all_bands: false` (default): flag if saturated in **any** band.
   - `require_all_bands: true`: flag only if saturated in **all** bands.
4. Affected sources are **flagged** with `excluded_saturation = True`. They remain in the catalog but are excluded from downstream fitting.

After both crop and saturation filtering, a composite flag `excluded_any = excluded_crop | excluded_saturation` is computed. The log reports the number of active (non-excluded) sources.

### 1.8 Render Overlay Plot

If `crop.overlay_catalog: true`:

- Render the post-crop white stack with source positions color-coded by `TYPE`:
  - **Cyan circles** — `STAR`
  - **Magenta circles** — `GAL`, `EXP`, `DEV`, `SERSIC`
  - **Yellow squares** — unknown/missing type (labeled with fallback model)
  - **Red X markers** — saturation-excluded sources
  - **Gray markers** (legend only) — crop-excluded sources, NaN/out-of-bounds sources

### 1.9 Timing Summary

The stage logs a timing breakdown:

```
load_inputs timing [s]: prep=X.XX white=X.XX crop=X.XX sat=X.XX overlay=X.XX total=X.XX
```

---

## Stage 1b: Augment Catalog with Gaia Sources (if `zp.enabled`)

**Function:** `augment_catalog_with_gaia()` in `zp.py`

Matches GaiaXP synphot sources against the input catalog and optionally injects unmatched Gaia sources as new rows for ZP calibration.

### Augmentation Flow

1. Load GaiaXP synphot CSV. Filter by `zp.gaia_mag_min <= phot_g_mean_mag <= zp.gaia_mag_max`.
2. Project Gaia source RA/DEC to pixel coordinates using WCS.
3. Compute a square bounding box around **active (non-excluded)** input catalog sources. Expand to at least `zp.min_box_size_pix x min_box_size_pix`. Shift to keep square at image edges; clamp to crop bounds if the image is smaller. Excluded sources (crop/saturation) are not used for box computation to prevent out-of-bounds coordinates from inflating the box.
4. Filter Gaia sources to those within the bounding box.
5. RA/DEC match Gaia sources against the input catalog (using `zp.match_radius_arcsec`). Tag matched original sources with `gaia_source_id`. Backfill missing `FLUX_{band}` values for matched sources from Gaia synphot magnitudes.
6. If `zp.inject_gaia_sources: true` (default): Remove already-matched Gaia sources from the injection pool. Apply saturation filtering (same config as `source_saturation_cut`) to remaining Gaia sources. Create new catalog rows for unmatched Gaia sources: `ID=gaia_{source_id}`, `TYPE=STAR`, `FLUX_{band}` from synphot magnitudes (`10^((zp_ref - mag_{band}) / 2.5)`). Concatenate original catalog + new Gaia rows.
7. If `zp.inject_gaia_sources: false`: Skip injection entirely. Only the matching (step 5) and its backfill are performed. No new rows are added. Use this when you already have reference sources in the input catalog.
8. Save the augmented catalog as `ZP/{name}_with_Gaia.csv`.
9. Generate augmentation overlay plot. Excluded sources (crop/saturation) are overlaid on the plot: red X for saturation-excluded, orange X for crop-excluded.

### Bounding Box Logic

- The box is the tightest square containing all **active (non-excluded)** catalog sources, expanded to at least `min_box_size_pix` on each side.
- When the box hits an image edge, it shifts to maintain the square shape.
- If the image is smaller than `min_box_size_pix` in either dimension, the box is clamped to the image bounds.

---

## Stage 2: Build ePSF

**Function:** `build_epsf_from_config()` → `epsf.build_epsf()`

Constructs an empirical PSF for each band in each spatial grid cell.

### Grid Structure

The image is divided into `epsf_ngrid x epsf_ngrid` cells. For each cell and each band:

1. **SEP detection:** Run `sep.extract()` on the cell subimage to find all sources.
2. **Star selection:** Combine GaiaXP seeds (if enabled) with SEP detections:
   - GaiaXP sources are projected to pixel coordinates and magnitude-filtered.
   - Candidates are checked for: edge proximity, saturation, roundness, minimum separation, SNR.
   - Up to `max_stars` are selected per cell.
3. **Local background correction (optional, enabled by default):**
   - Estimate/subtract a 2D SEP background map per ePSF cell.
   - Subtract per-star local annulus background from extracted star cutouts using sigma-clipped annulus median.
   - Annulus radii are automatically clamped to the stamp size (over-large radius values are safely clipped).
4. **ePSF building:** Use `photutils.EPSFBuilder` to construct the ePSF from selected star cutouts.
5. **Normalization and cropping:** The ePSF is normalized to unit sum and center-cropped to `final_psf_size`.

Per patch, additional diagnostics are saved:
- `background_diagnostics.png` (raw patch / background map / background-subtracted patch)
- `star_local_background_diagnostics.png` (per-star raw/annulus/bg-sub stamps; one row per used star)
- `epsf_growth_curve.png` (encircled-energy curve)
- `epsf_residual_diagnostics.png` (median star/model/residual + normalized residual histogram)

Diagnostic generation can be controlled with:
- `epsf.save_patch_background_diagnostics`
- `epsf.save_star_local_background_diagnostics`
- `epsf.diagnostics_show_colorbar`

For `star_local_background_diagnostics.png`, if used stars have missing or duplicate `id_label`, the plot is skipped with a warning (pipeline continues; ePSF build is not failed).

### ePSF Cell Activity Filtering

If `epsf.skip_empty_epsf_patches: true`:

- Active ePSF cells are computed from **non-excluded** input catalog source positions (i.e. sources where `excluded_any = False`).
- Only cells containing at least one source are processed.
- Downstream patch definitions are also restricted to active cells.
- This significantly reduces computation in sparse fields.

### Parallel Band Execution

If `epsf.parallel_bands: true`, bands are processed concurrently using `epsf.max_workers` workers.

In interactive terminals (TTY), live per-worker progress lines are displayed, showing per-band completion, rate, and ETA. When a worker finishes one band, it picks up the next unprocessed band.

---

## Stage 3: Build Patches

**Function:** `build_patches_from_config()` → `patches.build_patches()`

Defines the spatial decomposition of the image into fitting patches.

### Patch Geometry

Each active ePSF cell is subdivided into `ngrid x ngrid` patches. Each patch has:

- **Base region:** The core area where sources are assigned.
- **ROI (region of interest):** The base region expanded by a halo of `halo_pix` pixels on each side, clamped to image bounds. The halo ensures that source models near patch edges have sufficient image context.

```
halo_pix = max((final_psf_size - 1) / 2 + 2, halo_pix_min)
```

### Output

- `patches.csv` — Tabular patch definitions.
- `patches.json` — JSON patch definitions (used by subsequent stages).

---

## Stage 4: Build Patch Inputs

**Function:** `build_patch_inputs_from_config()` → `patch_inputs.build_patch_inputs()`

Creates self-contained data payloads for each patch.

### Source Assignment

1. Filter to **active (non-excluded)** sources only — sources with `excluded_any = True` are not assigned to any patch.
2. Project active source RA/DEC to pixel coordinates in the white-stack frame.
3. For each patch, select sources whose pixel positions fall within the patch's **base** region.
4. Compute patch-local pixel coordinates: `x_pix_patch = x_pix_white - x0_roi`.

### Payload Contents

Each payload (`*.pkl.gz`) contains:

- **Per-band image cutouts:** Scaled image, total sigma, sky sigma, bad mask, scaling factor, file path.
- **Source sub-catalog:** All input catalog columns for sources in this patch, plus computed pixel coordinates.
- **Patch metadata:** Tags, grid indices, bounding boxes, source count.

### Filtering

If `patch_inputs.skip_empty_patch: true`:

- Patches with zero sources are not written to disk and not queued for fitting.

### Combination Behavior

| `skip_empty_epsf_patches` | `skip_empty_patch` | Effect |
|--------------------------|-------------------|--------|
| true | true | Most aggressive pruning: skip empty ePSF cells AND empty patches within active cells. |
| true | false | Skip empty ePSF cells, but write/run empty patches within active cells. |
| false | true | Process all ePSF cells, but skip writing empty patches. |
| false | false | Process everything (most expensive). |

---

## Stage 5: Run Patch Subprocesses

**Function:** `run_patch_subprocesses()` → `run_patches.run_subprocesses()`

Launches independent Python subprocesses to fit each patch.

### Fitting Modes

The fitting behavior is controlled by `patch_run.enable_multi_band_simultaneous_fitting`:

- **`true` (default) — Multi-band simultaneous fitting:** All bands are loaded into a single Tractor instance. The optimizer adjusts positions and morphology (shared across bands) and per-band fluxes simultaneously in one optimization run. This leverages cross-band information for tighter constraints.
- **`false` — Single-band independent fitting:** Each band is fitted independently in a separate Tractor instance. Position, morphology, and flux can all differ per band. All bands start from the same initial guesses (from the shared input catalog), but fitted values diverge independently. Bands are processed serially within each patch subprocess; parallelism occurs at the patch level.

### Per-Patch Fitting Flow

For each patch subprocess:

1. **Load payload** from `*.pkl.gz`.
2. **Build PSF** for each band:
   - Look for `epsf.npy` at the expected path for this ePSF cell and band.
   - If present and `nstars_used >= min_epsf_nstars_for_use`: use the ePSF (pixelized, hybrid, or Gaussian mixture).
   - Otherwise: use the fallback PSF model (Moffat, NCircularGaussian, or GaussianMixture from synthetic stamp).
3. **Build Tractor images:** Wrap each band's cutout as a Tractor `Image` with appropriate PSF, inverse-variance, and sky model.
4. **Initialize source catalog:**
   - Assign Tractor source model based on `TYPE` (or fallback).
   - Initialize fluxes from `FLUX_{band}` or aperture photometry.
   - Initialize galaxy shape from `ELL`, `THETA`, `Re` (or defaults).
5. **Optimize:**
   - *Multi-band mode:* Run the `ConstrainedOptimizer` on a single Tractor (all bands) for up to `n_opt_iters` iterations, stopping early if `dlnp < dlnp_stop`.
   - *Single-band mode:* For each band, create a separate Tractor (one image), build a fresh catalog from the same input, and run the optimizer independently.
6. **Extract results:** Record fitted positions, fluxes, flux errors, morphology parameters, and convergence diagnostics. In single-band mode, all fitted parameters are stored per band (e.g. `Re_{band}_fit`, `THETA_{band}_fit`); source type (`stype_fit`) is shared.
7. **Generate diagnostics:** Cutout montages, patch overview plot. In single-band mode, each band's model is rendered from its own fitted Tractor, and each band column in the montage shows that band's own fitted position.

### Progress Display

- **TTY:** In-place progress line showing done/total, running count, fail count, rate, and ETA.
- **Non-TTY:** Periodic heartbeat log messages.

### Resume Mode

If `patch_run.resume: true`:

- Patches whose output directories already exist are skipped.
- This allows resuming interrupted runs without re-fitting completed patches.
- **Warning:** If you change config parameters and re-run with resume, old patches retain their old parameters.

---

## Stage 6: Merge

**Function:** `merge_results()` → `merge.merge_catalogs()`

Combines all per-patch fit results into a single catalog.

### Merge Flow

1. Read the base catalog (the augmented catalog if ZP is enabled, otherwise the original input catalog). This catalog already contains `excluded_crop`, `excluded_saturation`, and `excluded_any` flag columns set during the load stage.
2. Compute `excluded_reason` from the flag columns.
3. Collect all `*_cat_fit.csv` files matching `merge.pattern`.
4. From each patch catalog, keep only fit-specific columns (those not already in the base catalog, plus the merge key).
5. Concatenate all patch catalogs.
6. Check for duplicate merge keys. Duplicates are dropped (keeping first occurrence) with a warning; this can occur for sources on patch boundaries.
7. Left-join the base catalog with fit results.
8. Optionally compute sky coordinates from fitted pixel positions using WCS. In multi-band mode: `RA_fit`/`DEC_fit` from `x_pix_white_fit`/`y_pix_white_fit`. In single-band mode: `RA_{band}_fit`/`DEC_{band}_fit` from `x_pix_white_{band}_fit`/`y_pix_white_{band}_fit` for each band.
9. Write the final catalog.

### Exclusion Columns

The merge adds four exclusion tracking columns:

- `excluded_crop` (bool)
- `excluded_saturation` (bool)
- `excluded_any` (bool)
- `excluded_reason` (string): `"crop"`, `"saturation"`, `"crop+saturation"`, or `""`.

See [Outputs](outputs.md) for complete column documentation.

---

## Stage 7: Compute Zero-Point Calibration (if `zp.enabled`)

**Function:** `compute_zp()` in `zp.py`

Derives per-band zero-point from Gaia-matched stars and applies calibrated AB magnitudes to all sources.

### Why ZP Calibration Is Needed

Images are scaled to a common nominal ZP (`zp_ref`, default 25.0) using the `ZP_AUTO` header keyword. However, `ZP_AUTO` is an approximation from prior photometric calibration and may not exactly match the true zero-point. Additionally, the Tractor's PSF-fitting photometry differs methodologically from whatever produced `ZP_AUTO` (e.g. aperture photometry). The ZP calibration stage measures the actual zero-point of the Tractor flux system by comparing fitted fluxes of Gaia-matched stars against their known GaiaXP synphot magnitudes.

The resulting `ZP_median` per band is the true ZP of the **scaled** flux system. The offset `ZP_median - zp_ref` reflects primarily the error in `ZP_AUTO`.

!!! warning "Do not compare ZP_median directly to ZP_AUTO"
    `ZP_median` lives in the scaled flux system (nominal ZP = `zp_ref`), while `ZP_AUTO` lives in the original unscaled system. Since each band's `ZP_AUTO` differs slightly from `zp_ref`, the difference `ZP_median - ZP_AUTO` conflates the practical photometric error with the scaling offset (`zp_ref - ZP_AUTO`). The meaningful diagnostic is `ZP_median - zp_ref`, which isolates the true photometric error. Only if `ZP_AUTO` happened to equal `zp_ref` exactly (no scaling applied) would `ZP_median - ZP_AUTO` directly reflect the practical error.

### ZP Computation Flow

1. Read the merged catalog. Identify Gaia-matched sources by non-empty `gaia_source_id`.
2. Exclude non-converged sources from the ZP star pool:
   - Multi-band mode: filter upfront using `opt_converged`.
   - Single-band mode: filter per-band using `opt_converged_{band}`.
3. For each band:
   a. Compute per-star ZP: `ZP_i = mag_{band}_gaia + 2.5 * log10(FLUX_{band}_fit)`.
   b. Propagate flux error: `ZP_err_i = (2.5/ln10) * (FLUXERR/FLUX)`.
   c. Apply MAD-based iterative sigma clipping (`zp.clip_sigma`, `zp.clip_max_iters`).
   d. Compute weighted median ZP (weighted by `1/ZP_err^2`) and MAD-based error.
   e. Apply calibrated magnitudes to all sources: `MAG_{band}_fit = ZP_median - 2.5*log10(FLUX)`, `MAGERR_{band}_fit = sqrt((flux_err_term)^2 + ZP_err^2)`.
4. Save diagnostic plots and summary CSVs to the `ZP/` directory.
5. Overwrite the merged catalog with the additional `MAG_*_fit` and `MAGERR_*_fit` columns.

### Standalone ZP Re-run

`tract7dt compute-zp --config ...` re-runs only steps 1-5 on the existing merged catalog. This is useful for adjusting `zp.clip_sigma` or `zp.clip_max_iters` without re-fitting. The augmentation parameters require a full pipeline re-run.

---

## Source Exclusion Tracking

Sources excluded from fitting are **flagged in-place** on the catalog — they are never physically removed. The exclusion flags are carried through the entire pipeline:

1. During crop filtering, out-of-crop sources are flagged with `excluded_crop = True`.
2. During saturation filtering, sources near saturated pixels are flagged with `excluded_saturation = True`.
3. A composite flag `excluded_any = excluded_crop | excluded_saturation` is computed immediately after filtering.
4. Flagged sources remain in the catalog through Gaia augmentation and are written to the augmented CSV.
5. At patch-building time, only active (`excluded_any = False`) sources are assigned to patches for fitting.
6. At merge time, the base catalog already contains all sources and their flags. Excluded sources appear with empty fit columns (`NaN`).
7. The `excluded_reason` column is computed during merge: `"crop"`, `"saturation"`, `"crop+saturation"`, or `""`.

This design ensures that the final catalog always contains **every** input source (plus any injected Gaia sources), making it straightforward to cross-match with external datasets. Excluded sources are clearly identifiable by their flag columns.

---

## Overlay TYPE Behavior

The overlay diagnostic plot categorizes sources by their `TYPE` column:

| Category | `TYPE` values | Marker |
|----------|---------------|--------|
| Star | `STAR` | Cyan circle |
| Galaxy | `GAL`, `EXP`, `DEV`, `SERSIC` | Magenta circle |
| Unknown | *(missing, blank, NaN, or any other value)* | Yellow square, labeled `UNKNOWN (N=...) (fallback=<gal_model>)` |

If the `TYPE` column is missing entirely from the catalog, all sources are categorized as Unknown.

---

## ePSF Progress Behavior

In parallel band mode with TTY output:

- One live line is rendered per worker slot, showing band name, progress bar, completion percentage, rate, and ETA.
- When a worker finishes one band, its slot is reassigned to the next unprocessed band.
- Slot notices indicate handoff transitions (e.g. `done m400 -> start m475`).

In non-TTY mode (e.g. batch jobs), periodic log messages report overall band progress.
