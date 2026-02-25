# Pipeline Behavior

This page describes the runtime behavior of each pipeline stage as implemented in code. Understanding this flow is essential for interpreting outputs and diagnosing issues.

## Pipeline Overview

The full pipeline (`tract7dt run`) executes up to eight sequential stages:

```
load_inputs → [augment_gaia] → build_epsf → build_patches → build_patch_inputs → run_patches → merge → [compute_zp]
```

Stages in brackets are conditional on `zp.enabled: true`.

Each stage can also be run independently via its own CLI command (see [Commands](commands.md)).

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
3. Sources outside the crop box are marked `excluded_crop = True` and removed from the working catalog.
4. Slice the white stack, per-band images, masks, sigma arrays, and WCS to the crop region.
5. Generate diagnostic plots (pre-crop, post-crop).

### 1.7 Apply Saturation Filter (if enabled)

If `source_saturation_cut.enabled: true`:

1. For each source, project RA/DEC to pixel coordinates.
2. Check a circular disk of radius `source_saturation_cut.radius_pix` pixels around each source.
3. If any pixel in the disk is flagged as saturated:
   - `require_all_bands: false` (default): remove if saturated in **any** band.
   - `require_all_bands: true`: remove only if saturated in **all** bands.
4. Removed sources are marked `excluded_saturation = True`.

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

Injects GaiaXP synphot sources into the input catalog so they can be fitted by the Tractor and used for zero-point calibration.

### Augmentation Flow

1. Load GaiaXP synphot CSV. Filter by `zp.gaia_mag_min <= phot_g_mean_mag <= zp.gaia_mag_max`.
2. Project Gaia source RA/DEC to pixel coordinates using WCS.
3. Compute a square bounding box around all original input catalog sources. Expand to at least `zp.min_box_size_pix x min_box_size_pix`. Shift to keep square at image edges; clamp to crop bounds if the image is smaller.
4. Filter Gaia sources to those within the bounding box.
5. RA/DEC match Gaia sources against the input catalog (using `zp.match_radius_arcsec`). Tag matched original sources with `gaia_source_id`. Backfill missing `FLUX_{band}` values for matched sources from Gaia synphot magnitudes.
6. Remove already-matched Gaia sources from the injection pool. Apply saturation filtering (same config as `source_saturation_cut`) to remaining Gaia sources.
7. Create new catalog rows for unmatched Gaia sources: `ID=gaia_{source_id}`, `TYPE=STAR`, `FLUX_{band}` from synphot magnitudes (`10^((zp_ref - mag_{band}) / 2.5)`).
8. Concatenate original catalog + new Gaia rows. Save as `ZP/{name}_with_Gaia.csv`.
9. Generate augmentation overlay plot.

### Bounding Box Logic

- The box is the tightest square containing all original catalog sources, expanded to at least `min_box_size_pix` on each side.
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
3. **ePSF building:** Use `photutils.EPSFBuilder` to construct the ePSF from selected star cutouts.
4. **Normalization and cropping:** The ePSF is normalized to unit sum and center-cropped to `final_psf_size`.

### ePSF Cell Activity Filtering

If `epsf.skip_empty_epsf_patches: true`:

- Active ePSF cells are computed from the input catalog source positions after crop/saturation filtering.
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

1. Project all source RA/DEC to pixel coordinates in the white-stack frame.
2. For each patch, select sources whose pixel positions fall within the patch's **base** region.
3. Compute patch-local pixel coordinates: `x_pix_patch = x_pix_white - x0_roi`.

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

1. Read the original input catalog.
2. Attach exclusion flags (crop, saturation) using the merge key.
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

Derives per-band zero-point from Gaia-matched stars and applies AB magnitudes to all sources.

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
   e. Apply to all sources: `MAG_{band}_fit = ZP - 2.5*log10(FLUX)`, `MAGERR_{band}_fit = sqrt((flux_err_term)^2 + ZP_err^2)`.
4. Save diagnostic plots and summary CSVs to the `ZP/` directory.
5. Overwrite the merged catalog with the additional `MAG_*_fit` and `MAGERR_*_fit` columns.

### Standalone ZP Re-run

`tract7dt compute-zp --config ...` re-runs only steps 1-5 on the existing merged catalog. This is useful for adjusting `zp.clip_sigma` or `zp.clip_max_iters` without re-fitting. The augmentation parameters require a full pipeline re-run.

---

## Source Exclusion Tracking

Sources filtered out before fitting are tracked throughout the pipeline and merged back into the final output:

1. During crop filtering, removed sources are flagged with `excluded_crop = True`.
2. During saturation filtering, removed sources are flagged with `excluded_saturation = True`.
3. The exclusion flags are keyed by `ID` (preferred) or `(RA, DEC)`.
4. At merge time, flags are attached to the original input catalog rows.
5. Composite columns `excluded_any` and `excluded_reason` are computed.

This design ensures that the final catalog is always the same length as the input catalog, making it straightforward to cross-match with external datasets.

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
