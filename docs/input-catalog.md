# Input Catalog Reference

This page is the definitive reference for the three input files consumed by `tract7dt`:

1. **Input source catalog** (`inputs.input_catalog`) — the CSV table of sources to be fit.
2. **Image list** (`inputs.image_list_file`) — the list of FITS image paths.
3. **GaiaXP synphot catalog** (`inputs.gaiaxp_synphot_csv`) — optional Gaia DR3 synthetic photometry for ePSF star selection.

Understanding the format requirements, column semantics, and physical units of these files is essential before running the pipeline.

---

## 1. Input Source Catalog

**Config key:** `inputs.input_catalog`

**Format:** CSV (comma-separated values), readable by `pandas.read_csv()`.

This is the primary science input: a table of astronomical sources to be modeled and fit by the Tractor. Each row represents one source. The pipeline projects source sky positions onto the image pixel grid, assigns each source to a spatial patch, initializes a Tractor source model (point source or galaxy), and fits that model simultaneously across all bands.

### 1.1 Required Columns

| Column | Type   | Unit / Convention | Description |
|--------|--------|-------------------|-------------|
| `RA`   | float  | **degrees** (ICRS) | Right Ascension of the source. |
| `DEC`  | float  | **degrees** (ICRS) | Declination of the source. |

!!! warning "Column name matching"
    Column lookup is **case-insensitive** (e.g. `ra`, `Ra`, `RA` are all accepted), but the pipeline emits a warning if the exact column names are not `RA` and `DEC`. For clarity and to avoid warnings, always use uppercase `RA` and `DEC`.

!!! warning "Coordinate system"
    RA/DEC are interpreted through the WCS of the input FITS images via `astropy.wcs.WCS.all_world2pix()`. The coordinate frame must match the WCS frame of your images — typically ICRS (J2000 FK5). If your catalog uses a different epoch or frame, you must preprocess it to match before running the pipeline.

**Validation rules:**

- If `RA` or `DEC` columns cannot be found (case-insensitive), the pipeline raises `ValueError`.
- Rows with `NaN` or non-finite RA/DEC are silently excluded from spatial assignment to patches, but are preserved in the final merged catalog (with empty fit columns).

### 1.2 Recommended Columns

These columns are not strictly required but significantly improve model initialization and diagnostic quality.

| Column | Type   | Unit / Convention | Description |
|--------|--------|-------------------|-------------|
| `ID`   | string | — | Unique source identifier. **Strongly recommended.** Used as the merge key when joining patch fit results back to the input catalog. If absent, the pipeline falls back to using `RA`+`DEC` as a composite merge key, which is fragile for sources at identical coordinates. IDs are treated as **format-free strings** — the pipeline reads and preserves them exactly as written in the CSV. Leading zeros, non-numeric characters, and arbitrary text are all preserved (e.g. `00019`, `1858283`, `my_star_alpha`). The ID is also used in cutout filenames (`src_{ID}.png`). |
| `TYPE` | string | One of: `STAR`, `GAL`, `EXP`, `DEV`, `SERSIC` | Morphological classification that determines the Tractor source model. See [Source Model Mapping](#14-source-model-mapping) below. |

### 1.3 Optional Columns

| Column | Type | Unit / Convention | Description |
|--------|------|-------------------|-------------|
| `FLUX_{band}` | float | **Scaled flux** (in the reference zeropoint system set by `image_scaling.zp_ref`, default 25.0 mag) | Per-band initial flux estimate. `{band}` must exactly match the `FILTER` keyword in the corresponding FITS header (e.g. `FLUX_m400`, `FLUX_m625`). When provided, these values seed the Tractor optimizer. When absent, the pipeline performs SEP aperture photometry on the images to initialize fluxes. |
| `ELL`   | float | Dimensionless, range [0, 1) | Ellipticity, defined as `1 - b/a` where `b/a` is the axis ratio. Used to initialize galaxy model shape. If absent, defaults to `0.2` (axis ratio `b/a = 0.8`). Only used for galaxy-type sources. |
| `THETA` | float | **degrees** | Position angle in **SExtractor convention**: measured counter-clockwise from the image y-axis (north in standard orientation). Internally converted to the Tractor's `EllipseESoft` parameterization. If absent, defaults to `0`. Only used for galaxy-type sources. |
| `Re`    | float | **pixels** | Effective (half-light) radius. Used to initialize the galaxy model size parameter. If absent, falls back to `patch_run.re_fallback_pix` (default: 3.0 pixels). Values are clipped to [0.3, 100.0] pixels internally. Only used for galaxy-type sources. |

### 1.4 Source Model Mapping

The `TYPE` column determines which Tractor source model is used for each source. Matching is **case-insensitive** (the value is uppercased internally).

| `TYPE` value | Tractor model | Description |
|-------------|---------------|-------------|
| `STAR`      | `PointSource`   | Unresolved point source. Only position and per-band fluxes are fit. |
| `EXP`       | `ExpGalaxy`     | Exponential-profile galaxy (Sersic n=1). Fits position, fluxes, effective radius, ellipticity, and position angle. |
| `DEV`       | `DevGalaxy`     | de Vaucouleurs-profile galaxy (Sersic n=4). Same free parameters as `EXP`. |
| `SERSIC`    | `SersicGalaxy`  | General Sersic profile. Fits all `EXP`/`DEV` parameters plus the Sersic index `n` (initialized to `patch_run.sersic_n_init`, default 3.0). |
| `GAL`       | (same as fallback) | Alias. Treated as unknown type; the fallback model is used. |
| *(missing, empty, or any other value)* | Fallback model | Determined by `patch_run.gal_model` config key (default: `exp`). Allowed values: `exp`, `dev`, `sersic`, `star`. |

!!! note "TYPE column entirely absent"
    If the `TYPE` column does not exist at all, **every** source is assigned the fallback model set by `patch_run.gal_model`. In overlay diagnostic plots, all such sources appear as yellow squares labeled `UNKNOWN (fallback=<model>)`.

!!! tip "Choosing a fallback model"
    For fields dominated by faint, unresolved sources, consider `patch_run.gal_model: "star"`. For deep extragalactic fields with mostly galaxies, `"exp"` is a common choice.

### 1.5 Band Consistency Check

If the input catalog contains **any** column whose name starts with `FLUX_` (e.g. `FLUX_m400`, `FLUX_m625`), the pipeline performs a **band consistency check**:

- It extracts band names from catalog column names: `{FLUX_m400 → m400, FLUX_m625 → m625, ...}`.
- It extracts band names from image `FILTER` headers: `{m400, m625, ...}`.
- If these two sets do not match exactly, the pipeline raises `RuntimeError` with a message listing which bands are catalog-only and which are image-only.

This prevents silent mismatches between your catalog flux columns and your actual images.

!!! tip "No FLUX columns? No check."
    If your catalog has no `FLUX_*` columns at all, this check is skipped and fluxes are initialized purely from aperture photometry on the images.

### 1.6 Flux Initialization Logic

For each source and each band, the Tractor optimizer is seeded with an initial flux value. The selection logic is:

1. If `FLUX_{band}` exists in the catalog **and** the value is finite and positive → use it.
2. Otherwise → perform circular aperture photometry on the scaled image using `sep.sum_circle()` with radius `patch_run.r_ap` (default: 5.0 pixels).
3. The result is clamped to a minimum floor of `patch_run.eps_flux` (default: 1e-4) to avoid zero-flux initialization failures.

### 1.7 Coordinate Projection

Source RA/DEC are projected to pixel coordinates using the WCS of the first loaded FITS image (`astropy.wcs.WCS.all_world2pix()`). The projection is done once during `load_inputs()` and reused for:

- Crop filtering (is the source inside the crop box?)
- Saturation filtering (is the source near saturated pixels?)
- Patch assignment (which spatial patch does the source belong to?)
- Overlay diagnostic plots

Sources whose RA/DEC are `NaN`, non-finite, or project outside the image bounds are tracked but do not participate in fitting. They still appear in the final merged catalog.

### 1.8 Minimal Example

A minimal input catalog CSV:

```csv
ID,RA,DEC,TYPE
1,34.40625,-5.22306,STAR
2,34.40750,-5.22100,EXP
3,34.40500,-5.22500,DEV
```

A more complete example with flux priors and shape parameters:

```csv
ID,RA,DEC,TYPE,FLUX_m400,FLUX_m475,FLUX_m550,FLUX_m625,ELL,THETA,Re
1,34.40625,-5.22306,STAR,150.3,200.1,250.5,180.2,,,
2,34.40750,-5.22100,EXP,50.0,60.0,70.0,55.0,0.3,45.0,5.2
3,34.40500,-5.22500,SERSIC,80.0,90.0,100.0,85.0,0.15,120.0,8.0
```

!!! note "Stars and shape columns"
    For `STAR`-type sources, `ELL`, `THETA`, and `Re` are ignored (point sources have no shape parameters). It is safe to leave them blank or `NaN` for stars.

### 1.9 Common Mistakes and Validation Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ValueError: input_catalog must have RA/DEC columns` | No column named `RA` or `DEC` (case-insensitive). | Rename your coordinate columns to `RA` and `DEC`. |
| `RuntimeError: Band mismatch between input catalog and image list` | `FLUX_*` column band names don't match image `FILTER` headers. | Ensure `FLUX_` suffixes exactly match FITS `FILTER` values, or remove all `FLUX_*` columns to use aperture-photometry initialization. |
| Warning: `input_catalog uses ra/dec columns; interpreting as Right Ascension / Declination` | Columns are lowercase `ra`/`dec` instead of `RA`/`DEC`. | Rename to uppercase. The pipeline works but warns. |
| Many sources with empty fit columns in output | Sources have `NaN` coordinates, project outside image bounds, or were flagged by crop/saturation filters. | Check `excluded_*` columns in the output. Verify coordinate frame matches WCS. |
| All sources labeled `UNKNOWN` in overlay plots | `TYPE` column is missing or contains unrecognized values. | Add a `TYPE` column, or accept the fallback model set by `patch_run.gal_model`. |
| `ValueError: Duplicate keys in patch results` | Multiple sources share the same `ID` (or same `RA`+`DEC` when `ID` is absent). | Ensure `ID` values are unique, or ensure no two sources share identical coordinates. |

---

## 2. Image List File

**Config key:** `inputs.image_list_file`

**Format:** Plain text file. One FITS image path per line.

### 2.1 Syntax

- **One path per line.** Each non-empty, non-comment line is interpreted as a FITS file path.
- **Comments:** Lines starting with `#` are ignored.
- **Blank lines** are ignored.
- **Relative paths** are resolved against the directory containing the YAML config file.
- **Absolute paths** are used as-is.

### 2.2 Example

```text
# 7DT UDS Field - All Bands
/data/7dt/7DT_UDS_m400.fits
/data/7dt/7DT_UDS_m475.fits
/data/7dt/7DT_UDS_m550.fits
/data/7dt/7DT_UDS_m625.fits
```

### 2.3 FITS Image Requirements

Each FITS file listed must satisfy specific header requirements. The pipeline reads the **primary HDU** (`hdul[0]`) and checks for the following:

#### Required FITS Header Keywords

| Keyword    | Type   | Unit / Convention | Description |
|-----------|--------|-------------------|-------------|
| `FILTER`  | string | — | Band/filter name (e.g. `m400`, `m625`). Must not be empty. This value is used as the band identifier throughout the pipeline and must match `FLUX_{band}` column suffixes if present in the input catalog. |
| `ZP_AUTO` | float  | **mag** (AB) | Photometric zero-point. Used to scale all images to a common reference zero-point (`image_scaling.zp_ref`, default 25.0). The scaling factor applied is `10^(-0.4 * (ZP_AUTO - zp_ref))`. |
| `SKYSIG`  | float  | **ADU** | Sky background noise (1-sigma) per pixel, in ADU. Must be positive and finite. Used to build the sky noise map and inverse-variance weights. |
| `EGAIN`   | float  | **e-/ADU** | Effective gain. Must be positive and finite. Used to compute source Poisson noise: `var_source = max(image, 0) / EGAIN`. |

#### Optional FITS Header Keywords

| Keyword    | Type   | Unit / Convention | Description |
|-----------|--------|-------------------|-------------|
| `SATURATE` | float | **ADU** | Saturation level of the detector. If present and finite, pixels at or above `SATURATE/2` are flagged as saturated in the bad-pixel mask. If absent, no saturation mask is built (pipeline warns and uses `inf`). |
| `PEEING`   | float | **pixels** | PSF FWHM in pixels. Used as fallback when the ePSF is not available for a given band/cell. Takes priority over `SEEING`. |
| `SEEING`   | float | **arcseconds** | Atmospheric seeing FWHM. Used as fallback PSF when `PEEING` is absent. Converted to pixels using the WCS pixel scale. |

#### WCS Requirements

Each image must contain a valid WCS (World Coordinate System) in its primary header, parsable by `astropy.wcs.WCS()`. All images must share a consistent, aligned WCS:

- **CRVAL** (reference position) must agree within `checks.wcs_tolerance.crval` (default: 1e-6 degrees).
- **CRPIX** (reference pixel) must agree within `checks.wcs_tolerance.crpix` (default: 1e-6 pixels).
- **CD/PC matrix** must agree within `checks.wcs_tolerance.cd` (default: 1e-9).
- **CDELT** must agree within `checks.wcs_tolerance.cdelt` (default: 1e-9 deg/pix).
- **CTYPE** must be identical across all images.

If `checks.require_wcs_alignment: true` (default) and any image fails this check, the pipeline raises `RuntimeError`.

#### Shape Requirement

If `checks.require_same_shape: true` (default), all images must have identical `(NAXIS2, NAXIS1)` dimensions. This is expected because 7DT images for a given tile are pre-registered and stacked to a common grid.

### 2.4 Image Scaling

All images are scaled to a common photometric reference system before any analysis. For each image:

```
scale_factor = 10^(-0.4 * (ZP_AUTO - zp_ref))
scaled_image = raw_image * scale_factor
```

Where `zp_ref` is set by `image_scaling.zp_ref` (default: 25.0). This means that in the scaled system:

- A source with flux `F` (in scaled counts) has **approximate** magnitude: `mag ≈ -2.5 * log10(F) + zp_ref`. This is exact only if `ZP_AUTO` in the FITS header perfectly represents the true zero-point. In practice, `ZP_AUTO` has errors, so the pipeline's ZP calibration stage (when `zp.enabled`) derives the actual ZP from Gaia-matched stars and applies calibrated `MAG_{band}_fit` columns.
- All `FLUX_*` values in the input catalog (if provided) should be in this same scaled system.
- All fitted flux values in the output catalog are in this system.

### 2.5 Noise Model

For each pixel in each band, the total noise variance is:

```
var_total = SKYSIG^2 + max(raw_image, 0) / EGAIN
sigma_total = sqrt(var_total) * scale_factor
```

The pipeline uses the **sky-only sigma** (not total sigma) as the default noise estimate for Tractor fitting (`sigma_sky_scaled`), which means:

```
sigma_sky_scaled = SKYSIG * scale_factor
invvar = 1 / sigma_sky_scaled^2   (for good pixels)
invvar = 0                         (for bad/saturated pixels)
```

### 2.6 Common FITS Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: Missing FILTER in <file>` | The FITS header has no `FILTER` keyword. | Add `FILTER` keyword to the header. |
| `RuntimeError: Empty FILTER in <file>` | `FILTER` header value is blank or whitespace. | Set `FILTER` to the band name string. |
| `RuntimeError: Missing ZP_AUTO in <file>` | No `ZP_AUTO` in header. | Run photometric calibration and write `ZP_AUTO`. |
| `RuntimeError: Bad/Missing SKYSIG in <file>` | `SKYSIG` is absent, non-finite, or ≤ 0. | Compute and write a valid `SKYSIG` value. |
| `RuntimeError: Bad/Missing EGAIN in <file>` | `EGAIN` is absent, non-finite, or ≤ 0. | Write a valid `EGAIN` value (typically from detector specs). |
| `RuntimeError: WCS mismatch with reference: <file>` | WCS does not agree with the first image within tolerances. | Re-register images to a common WCS, or relax `checks.wcs_tolerance`. |
| `RuntimeError: Image shape mismatch: <file>` | Array dimensions differ from the first image. | Ensure all images are resampled to the same pixel grid. |
| Warning: `SATURATE missing in <file>; using inf.` | No `SATURATE` keyword. | Add `SATURATE` if you want saturation masking; otherwise this warning is harmless. |

---

## 3. GaiaXP Synphot Catalog

**Config key:** `inputs.gaiaxp_synphot_csv`

**Format:** CSV, readable by `pandas.read_csv()`.

This catalog provides Gaia DR3 XP synthetic photometry for stars in the field. It is used during ePSF (effective Point Spread Function) construction to identify high-quality PSF star candidates. The pipeline cross-matches these positions with SEP source detections in the image to select clean, unsaturated, isolated stars for ePSF building.

### 3.1 Required Columns

| Column | Type | Unit / Convention | Description |
|--------|------|-------------------|-------------|
| `ra`   | float | **degrees** (ICRS) | Gaia DR3 right ascension. **Note: lowercase.** |
| `dec`  | float | **degrees** (ICRS) | Gaia DR3 declination. **Note: lowercase.** |
| `mag_{band}` | float | **mag** (AB, synthetic) | Synthetic magnitude in the band matching the FITS `FILTER` name. One column per band. For example, if your images have `FILTER=m400` and `FILTER=m625`, this catalog needs columns `mag_m400` and `mag_m625`. |

!!! warning "Column names are lowercase"
    Unlike the input source catalog (which uses uppercase `RA`/`DEC`), the GaiaXP catalog uses **lowercase** `ra` and `dec`. This reflects Gaia catalog conventions.

### 3.2 Example

```csv
ra,dec,mag_m400,mag_m475,mag_m550,mag_m625
34.405,−5.223,18.2,17.8,17.5,17.3
34.410,−5.225,19.1,18.7,18.4,18.2
```

### 3.3 Magnitude Filtering

Only GaiaXP sources within the configured magnitude range are used as PSF star seeds:

- `epsf.gaia_mag_min` (default: 10.0 mag) — reject sources brighter than this (likely saturated).
- `epsf.gaia_mag_max` (default: 30.0 mag) — reject sources fainter than this (too faint for PSF).

### 3.4 How It Is Used

The GaiaXP catalog is used only during the ePSF building stage (`run-epsf` or the ePSF step within `run`). The flow is:

1. For each band, look for column `mag_{band}` in the GaiaXP CSV.
2. Filter to sources with magnitudes within `[gaia_mag_min, gaia_mag_max]`.
3. Project GaiaXP RA/DEC to pixel coordinates using image WCS.
4. For each ePSF grid cell, gather GaiaXP seeds that fall within the cell.
5. Depending on `epsf.psfstar_mode`:
   - `"gaia"` — use only GaiaXP positions.
   - `"sep+gaia"` (default) — use GaiaXP positions as priority seeds, supplemented by SEP-detected sources.
   - `"sep"` — ignore GaiaXP entirely, use only SEP detections.
6. If `epsf.gaia_snap_to_sep: true` (default), GaiaXP seed positions are refined by snapping to the nearest SEP detection within `epsf.gaia_match_r_pix` pixels.
7. Each candidate undergoes SNR and saturation checks before being accepted.

### 3.5 When It Can Be Omitted

You can omit the GaiaXP catalog by either:

- Setting `epsf.use_gaiaxp: false` in the config.
- Setting `epsf.psfstar_mode: "sep"` (SEP-only star selection).
- Pointing the path to a non-existent file (the pipeline warns but does not crash).

In these cases, PSF star selection relies entirely on automated SEP source detection and selection (roundness filter, brightness ranking, saturation avoidance).

### 3.6 Generating a GaiaXP Synphot Catalog

The GaiaXP synphot catalog is typically generated by:

1. Querying Gaia DR3 sources within the tile footprint (e.g. via TAP, `astroquery`, or a local copy).
2. Selecting sources with XP spectra available.
3. Computing synthetic photometry through the 7DT filter response curves.
4. Writing the result as a CSV with the required columns.

This catalog is field-specific and must be regenerated for each new tile/pointing.

---

## 4. Summary: Input File Checklist

Before running the pipeline, verify:

- [ ] **Input catalog CSV** exists at the path specified in `inputs.input_catalog`.
- [ ] Catalog has `RA` and `DEC` columns (uppercase recommended).
- [ ] Catalog has a unique `ID` column (strongly recommended).
- [ ] If `FLUX_*` columns exist, their band suffixes exactly match FITS `FILTER` values.
- [ ] RA/DEC coordinates are in the same frame as the image WCS (typically ICRS).
- [ ] **Image list file** exists at `inputs.image_list_file`.
- [ ] All listed FITS files exist and are accessible.
- [ ] All FITS files have `FILTER`, `ZP_AUTO`, `SKYSIG`, `EGAIN` headers.
- [ ] All FITS files have valid, consistent WCS.
- [ ] All FITS files have the same pixel dimensions.
- [ ] **GaiaXP CSV** exists at `inputs.gaiaxp_synphot_csv` (if using GaiaXP-based ePSF).
- [ ] GaiaXP CSV has lowercase `ra`, `dec`, and `mag_{band}` columns for each band.

---

## 5. Physical Units Summary

This table collects all physical units in one place for quick reference.

| Quantity | Unit | Where used |
|----------|------|------------|
| RA / DEC (source catalog) | degrees (ICRS) | `input_catalog` columns `RA`, `DEC` |
| RA / DEC (GaiaXP catalog) | degrees (ICRS) | `gaiaxp_synphot_csv` columns `ra`, `dec` |
| Flux (`FLUX_{band}`) | scaled counts (ZP = `zp_ref`) | `input_catalog` optional columns |
| Ellipticity (`ELL`) | dimensionless, 1 − b/a | `input_catalog` optional column |
| Position angle (`THETA`) | degrees (SExtractor convention: CCW from y-axis) | `input_catalog` optional column |
| Effective radius (`Re`) | pixels | `input_catalog` optional column |
| `ZP_AUTO` | magnitudes (AB) | FITS header |
| `SKYSIG` | ADU (per pixel) | FITS header |
| `EGAIN` | electrons per ADU (e⁻/ADU) | FITS header |
| `SATURATE` | ADU | FITS header |
| `PEEING` | pixels (FWHM) | FITS header |
| `SEEING` | arcseconds (FWHM) | FITS header |
| GaiaXP magnitudes (`mag_{band}`) | magnitudes (AB, synthetic) | `gaiaxp_synphot_csv` |
| WCS tolerance: `crval` | degrees | config `checks.wcs_tolerance.crval` |
| WCS tolerance: `crpix` | pixels | config `checks.wcs_tolerance.crpix` |
| WCS tolerance: `cd`, `cdelt` | degrees/pixel | config `checks.wcs_tolerance.cd`, `.cdelt` |
| Crop margin | pixels | config `crop.margin` |
| Saturation search radius | pixels | config `source_saturation_cut.radius_pix` |
| Aperture radius (flux init) | pixels | config `patch_run.r_ap` |
| Effective radius fallback | pixels | config `patch_run.re_fallback_pix` |
| Fallback PSF FWHM | pixels | config `patch_run.fallback_default_fwhm_pix` |
