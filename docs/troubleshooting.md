# Troubleshooting

This page covers common errors, warnings, and unexpected behaviors, organized by pipeline stage.

---

## Input Catalog Errors

### `ValueError: input_catalog must have RA/DEC columns`

**Cause:** The CSV file has no column named `RA` or `DEC` (case-insensitive search).

**Fix:** Rename your coordinate columns to `RA` and `DEC`. The pipeline accepts any case (e.g. `ra`, `Ra`, `RA`) but always warns if the exact names are not uppercase.

### `RuntimeError: Band mismatch between input catalog and image list`

**Cause:** Your catalog contains `FLUX_*` columns whose band suffixes don't exactly match the `FILTER` keyword values in your FITS images. For example, the catalog has `FLUX_m400` but the image has `FILTER = M400` (different case or extra whitespace).

**Fix:**

- Check that `FLUX_` suffixes match FITS `FILTER` values exactly (case-sensitive after stripping whitespace).
- If you don't need flux priors, remove all `FLUX_*` columns from the catalog. The pipeline will initialize fluxes from aperture photometry instead.

### `ValueError: Duplicate keys in patch results`

**Cause:** Multiple sources in the input catalog share the same merge key. If `ID` exists, duplicate `ID` values cause this. If `ID` is absent, duplicate `(RA, DEC)` pairs cause this.

**Fix:** Ensure `ID` values are unique. If using coordinate-based merging, ensure no two sources have identical RA and DEC.

### Warning: `input_catalog uses ra/dec columns`

**Cause:** The pipeline found coordinate columns but they are not exactly named `RA` and `DEC` (e.g. they are lowercase `ra`/`dec`).

**Fix:** Rename to uppercase `RA`/`DEC`. The pipeline works either way, but the warning indicates a potential naming convention issue.

---

## FITS Image Errors

### `RuntimeError: Missing FILTER in <file>`

**Fix:** Add a `FILTER` keyword to the FITS header with the band name string.

### `RuntimeError: Empty FILTER in <file>`

**Fix:** Set the `FILTER` header value to a non-empty band name.

### `RuntimeError: Missing ZP_AUTO in <file>`

**Fix:** Run photometric calibration on the image and write the `ZP_AUTO` keyword.

### `RuntimeError: Bad/Missing SKYSIG in <file>`

**Fix:** Compute the sky background noise (1-sigma, in ADU) and write it as `SKYSIG`. The value must be positive and finite.

### `RuntimeError: Bad/Missing EGAIN (e-/ADU) in <file>`

**Fix:** Write the effective gain in electrons per ADU as `EGAIN`. This is typically a detector/instrument property.

### `RuntimeError: Image shape mismatch`

**Cause:** Not all images have the same pixel dimensions.

**Fix:** Ensure all images are resampled/registered to the same pixel grid. All images for a given tile should share identical `(NAXIS2, NAXIS1)`.

### `RuntimeError: WCS mismatch with reference`

**Cause:** The WCS of one image does not agree with the first image within the configured tolerances.

**Fix:**

- Re-register images to a common WCS.
- Or relax the tolerance values in `checks.wcs_tolerance.*` (not recommended unless you understand the implications).
- Or disable the check: `checks.require_wcs_alignment: false` (not recommended).

### Warning: `SATURATE missing in <file>; using inf`

**Cause:** The FITS header lacks a `SATURATE` keyword.

**Fix:** This is often harmless — it means no saturation mask will be built for this image. If you need saturation filtering, add the `SATURATE` keyword to the header.

---

## Load Stage Performance

### Long pauses in `load_inputs`

Expected heavy steps:

- **Per-frame FITS prep:** Loading large images, computing masks and noise arrays. Scales with number of bands.
- **White-stack build:** Inverse-variance weighted coaddition. Memory-intensive.
- **Diagnostic plot rendering:** Large PNG files for white-stack and overlay plots.

Use the timing summary to locate bottlenecks:

```
load_inputs timing [s]: prep=X.XX white=X.XX crop=X.XX sat=X.XX overlay=X.XX total=X.XX
```

**Mitigation strategies:**

- Increase `performance.frame_prep_workers` (but watch memory).
- Increase `performance.white_stack_workers`.
- Reduce diagnostic plot resolution: `crop.display_downsample`, `crop.post_crop_display_downsample`, `crop.overlay_downsample_full`.

---

## ePSF Issues

### Few or no ePSF outputs

**Check:**

- `epsf.skip_empty_epsf_patches: true` may be intentionally pruning empty cells. This is expected in sparse fields.
- Source density after crop and saturation flagging may be very low.
- ePSF selection thresholds may be too aggressive:
  - `epsf.thresh_sigma` (detection threshold) — lowering it finds fainter candidates.
  - `epsf.minarea` (minimum detection area) — lowering it accepts smaller detections.
  - `epsf.gaia_snr_min` (SNR requirement) — lowering it accepts fainter candidates.
  - `epsf.q_range` (roundness filter) — widening it accepts less round sources.

### ePSF quality warnings (low star count)

If the log shows `low-star ePSF (nstars_used=N < min_epsf_nstars_for_use=M)`:

- The ePSF was built but from too few stars to be reliable.
- The pipeline will use the fallback PSF model for affected bands in that cell.
- Reduce `patch_run.min_epsf_nstars_for_use` if you are willing to accept lower-quality ePSFs, or increase `epsf.max_stars` and relax selection criteria.

### GaiaXP catalog issues

If the log shows `GaiaXP catalog not found`:

- Check the path in `inputs.gaiaxp_synphot_csv`.
- If you don't have a GaiaXP catalog for this field, set `epsf.use_gaiaxp: false` or `epsf.psfstar_mode: "sep"`.

If GaiaXP is loaded but no stars are selected:

- Check that the `mag_{band}` column names match the FITS `FILTER` values.
- Check that `epsf.gaia_mag_min` / `epsf.gaia_mag_max` bracket the expected magnitude range.

---

## Patch Run Issues

### Many empty patch runs

**Enable pruning:**

```yaml
patch_inputs:
  skip_empty_patch: true

epsf:
  skip_empty_epsf_patches: true
```

This prevents writing and running patches that contain no sources.

### Optimizer convergence warnings

If many patches show `opt_hit_max_iters: true`:

- Increase `patch_run.n_opt_iters` (at the cost of runtime).
- Relax `patch_run.dlnp_stop` (accept a larger dlnp change as "converged").
- Check if the PSF model is appropriate — a poor PSF can prevent convergence.

### Subprocess crashes

Check `<tag>.runner.log` for the full subprocess output. Common causes:

- Out-of-memory (too many sources in a patch, or too-large cutout size).
- Missing ePSF files (check the `epsf_root` path).
- Tractor internal errors (rare; usually indicates a degenerate source configuration).

---

## Overlay Plot Confusion

### All sources labeled UNKNOWN

**Cause:** The `TYPE` column is missing from the input catalog, or all values are unrecognized.

**Fix:** Add a `TYPE` column with values from: `STAR`, `EXP`, `DEV`, `SERSIC`. Or accept the fallback: all sources will be fit as `patch_run.gal_model`.

### Overlay TYPE grouping reference

| Category | Matched `TYPE` values |
|----------|----------------------|
| STAR (cyan) | `STAR` |
| GAL (magenta) | `GAL`, `EXP`, `DEV`, `SERSIC` |
| UNKNOWN (yellow) | Anything else, including empty, `NaN`, `NULL` |

---

## Merge Issues

### Rows with no fit values in the output

This can be normal. Sources that were **flagged** as excluded are preserved in the final catalog but not assigned to any patch, so their fit columns are empty. Inspect:

- `excluded_crop` — was the source outside the crop region?
- `excluded_saturation` — was the source near saturated pixels?
- `excluded_any` — was the source excluded for any reason?
- `excluded_reason` — human-readable reason: `"crop"`, `"saturation"`, `"crop+saturation"`, or `""`.

If `excluded_any` is false but fit columns are still empty, the source may have had `NaN` coordinates or projected outside image bounds.

### ID column read as numeric (leading zeros lost)

**Cause:** When per-patch fit CSVs are read, pandas may infer an all-numeric `ID` column as `int64`, stripping leading zeros (e.g. `00101` → `101`). This causes a merge key mismatch.

**Status:** This is handled automatically. The pipeline reads `ID` as string dtype in all CSV reads (both base and patch-fit catalogs), preserving the original format. IDs are format-free strings — `00019`, `1858283`, and `star_alpha` all work correctly.

### `download-sample` logging differs from pipeline logging

This is expected:

- Pipeline commands (`run`, `run-epsf`, etc.) apply the YAML `logging.*` configuration. By default, logs are written to `{work_dir}/{command}.log` (e.g. `run.log`, `merge.log`).
- `download-sample` runs without `--config`, so it uses its own independent logging setup.

### Log file location

With the default `logging.file: "auto"`, each command writes its log to `{work_dir}/{command}.log`. To change this:

- Set `logging.file: "my_custom.log"` — resolved relative to `work_dir`.
- Set `logging.file: null` — disable file logging entirely (console only).

---

## General Tips

### Checking available bands

To see which bands/filters are in your images without running the full pipeline:

```bash
for f in /path/to/images/*.fits; do
  python -c "from astropy.io import fits; print('$f', fits.getheader('$f')['FILTER'])"
done
```

### Verifying catalog-image band alignment

Check that your `FLUX_*` column suffixes match FITS `FILTER` values:

```python
import pandas as pd
from astropy.io import fits

cat = pd.read_csv("input_catalog.csv")
cat_bands = {c.replace("FLUX_", "") for c in cat.columns if c.startswith("FLUX_")}

# Compare with FITS FILTER values
for f in ["image1.fits", "image2.fits"]:
    print(f, fits.getheader(f)["FILTER"])
```

### Inspecting exclusion statistics

After a pipeline run:

```python
import pandas as pd

df = pd.read_csv("output_catalog.csv")
print("Total sources:", len(df))
print("Excluded (crop):", df["excluded_crop"].sum())
print("Excluded (saturation):", df["excluded_saturation"].sum())
print("Excluded (any):", df["excluded_any"].sum())
print("Fitted:", df["opt_converged"].notna().sum())
print("Converged:", df["opt_converged"].sum())
print("Hit max iters:", df["opt_hit_max_iters"].sum())
```
