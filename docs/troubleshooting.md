# Troubleshooting

## Long pauses in `load_inputs`

Expected heavy steps:

- per-frame FITS prep
- white-stack build
- large diagnostic PNG rendering

Use `load_inputs timing [s]: ...` summary to locate bottlenecks.

## WCS/shape validation failure

If enabled checks fail:

- confirm all images share aligned WCS
- confirm all image shapes match
- verify tolerance values in `checks.wcs_tolerance`

## Few/No ePSF outputs

Check:

- `epsf.skip_empty_epsf_patches` (may intentionally prune empty cells)
- source density after crop/saturation filters
- ePSF selection thresholds (`thresh_sigma`, `minarea`, etc.)

## Many empty patch runs

Enable:

- `patch_inputs.skip_empty_patch: true`

Also keep:

- `epsf.skip_empty_epsf_patches: true`

for earlier pruning.

## Confusing overlay source classes

Overlay type grouping:

- STAR
- GAL/EXP/DEV/SERSIC
- UNKNOWN (includes missing/blank/non-standard types)

Fitting fallback model for unknown/missing types is `patch_run.gal_model`.

## `download-sample-data` logging differs from pipeline logging

This is expected:

- pipeline commands apply YAML logging config
- `download-sample-data` runs without `--config`

## Merge rows with no fit values

This can be normal for excluded/unfitted sources.
Inspect:

- `excluded_crop`
- `excluded_saturation`
- `excluded_any`
- `excluded_reason`
