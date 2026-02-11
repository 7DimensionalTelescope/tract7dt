# Outputs

With default sample layout, `outputs.work_dir` contains:

- `EPSFs/` ePSF artifacts
- `patches/` patch definitions (`patches.csv`, `patches.json`)
- `patch_payloads/` patch payloads (`*.pkl.gz`)
- `outputs/` patch run outputs
- `output_catalog.csv` merged catalog
- `cropped_images/` white/crop/overlay diagnostics

## Patch Output Directory

Each patch output directory typically includes:

- `<tag>.runner.log` subprocess command + stdout/stderr capture
- `<tag>_cat_fit.csv` patch fit table
- `meta.json` patch metadata + PSF audit + optimizer summary
- optional cutout montages
- optional patch overview PNG

## Final Merged Catalog

Merge is left-join from original input catalog onto patch fit results.

As a result:

- rows filtered out before patch generation still appear in final output
- fit columns may be empty for excluded/unfitted rows

### Exclusion columns (new)

- `excluded_crop` (bool)
- `excluded_saturation` (bool)
- `excluded_any` (bool)
- `excluded_reason`:
  - `crop`
  - `saturation`
  - `crop+saturation`
  - empty string (not excluded)

### Optimizer/fit diagnostics (selected)

- `opt_converged`
- `opt_hit_max_iters`
- `opt_niters`
- PSF audit summary columns (fallback/low-star usage by band)

## Merge Logging

Merge stage logs:

- output path
- row/column count
- exclusion counts (if exclusion columns present)
- merge-missing summary
