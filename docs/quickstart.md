# Quick Start

This guide walks you through preparing inputs and running the pipeline for the first time.

## Prerequisites

- Python >= 3.10 with `tract7dt` installed (see [Installation](installation.md)).
- The Tractor and astrometry.net installed in your environment.
- A set of aligned, WCS-registered FITS images for your tile.
- An input source catalog in CSV format.

## 1) Generate a Config Template

```bash
# Write sample_config.yaml in current directory
tract7dt dump-config

# Write to explicit path
tract7dt dump-config --out my_config.yaml

# Overwrite existing file
tract7dt dump-config --force
```

This creates a fully commented YAML file with all available parameters and their defaults.

## 2) Prepare Your Input Files

### Input Source Catalog

Create a CSV file with at minimum `RA` and `DEC` columns (in degrees, ICRS). For best results, include `ID`, `TYPE`, and optionally `FLUX_{band}`, `ELL`, `THETA`, `Re` columns. See [Input Catalog Reference](input-catalog.md) for complete format documentation.

Minimal example:

```csv
ID,RA,DEC,TYPE
1,34.40625,-5.22306,STAR
2,34.40750,-5.22100,EXP
3,34.40500,-5.22500,DEV
```

### Image List

Create a text file with one FITS image path per line:

```text
# 7DT images for UDS field
/data/7dt/7DT_UDS_m400.fits
/data/7dt/7DT_UDS_m475.fits
/data/7dt/7DT_UDS_m550.fits
/data/7dt/7DT_UDS_m625.fits
```

Each FITS file must have these header keywords: `FILTER`, `ZP_AUTO`, `SKYSIG`, `EGAIN`. See [Input Catalog Reference](input-catalog.md#2-image-list-file) for full requirements.

### GaiaXP Synphot Catalog (optional)

If using GaiaXP-based PSF star selection, prepare a CSV with columns `ra`, `dec` (lowercase), and `mag_{band}` for each band.

## 3) Edit the Config

At minimum, update these paths in the YAML config:

```yaml
inputs:
  input_catalog: "/path/to/your/input_catalog.csv"
  image_list_file: "/path/to/your/image_list.txt"
  gaiaxp_synphot_csv: "/path/to/gaiaxp_synphot.csv"  # or remove if not using

outputs:
  work_dir: "/path/to/output/directory"
```

Review other settings based on your data. Key decisions:

- `crop.margin` — adjust if your image edges have different noise widths.
- `patch_run.gal_model` — set the fallback model if your catalog lacks a `TYPE` column.
- `image_scaling.zp_ref` — keep at 25.0 unless you have a specific reason to change it.

## 4) Run the Full Pipeline

```bash
tract7dt run --config /path/to/config.yaml
```

This executes up to eight stages sequentially: load, [Gaia augmentation], ePSF, patches, patch inputs, patch runs, merge, [ZP computation]. Stages in brackets run when `zp.enabled: true`.

## 5) Useful Partial Commands

Run individual stages when debugging or iterating:

```bash
# Load inputs, augment with Gaia (if ZP enabled), and build ePSF only
tract7dt run-epsf --config /path/to/config.yaml

# Load inputs, augment, and build patch geometry
tract7dt build-patches --config /path/to/config.yaml

# Load inputs, augment, build patches, and write patch payloads
tract7dt build-patch-inputs --config /path/to/config.yaml

# Run Tractor fitting on existing payloads (no reload)
tract7dt run-patches --config /path/to/config.yaml

# Load inputs, augment, and merge existing patch results into final catalog
tract7dt merge --config /path/to/config.yaml

# Re-compute ZP on existing merged catalog
tract7dt compute-zp --config /path/to/config.yaml
```

See [Commands](commands.md) for details on each command.

## 6) Sample Data for Test Runs

To test your installation with a small sample dataset:

```bash
# Download to default directory under cwd
tract7dt download-sample

# Download to explicit directory
tract7dt download-sample --dir /path/to/dataset

# Overwrite existing directory
tract7dt download-sample --dir /path/to/dataset --force
```

The sample dataset includes pre-registered FITS images and a matching input catalog. See [Sample Data](sample-data.md) for download behavior and terms.

## 7) Check Your Results

After the pipeline completes:

1. **Final catalog:** Check `output_catalog.csv` (or your configured `outputs.final_catalog`).
2. **Exclusion statistics:** Look at `excluded_crop`, `excluded_saturation`, `excluded_any` columns.
3. **Convergence:** Check `opt_converged` and `opt_hit_max_iters` columns.
4. **Diagnostics:** Browse `cropped_images/white_overlay.png` for a visual overview.
5. **Per-source results:** Check `outputs/<patch>/cutouts/` for individual source fits.
6. **Run log:** Check `{work_dir}/run.log` for the full pipeline log.
7. **Config audit:** Check `{work_dir}/config_used_*.yaml` for the exact config used.

See [Outputs](outputs.md) for complete documentation of all output columns and files.
