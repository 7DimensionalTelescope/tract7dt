# Configuration

The pipeline is controlled by one YAML file.

- Start from `tract7dt/data/sample_config.yaml`.
- Unknown keys are rejected.
- Missing keys fall back to defaults from `tract7dt/config.py`.

## Path Resolution Rules

- `inputs.*` relative paths resolve against the YAML file directory.
- `outputs.work_dir` resolves against YAML directory if relative.
- Other `outputs.*` relative paths resolve against `outputs.work_dir`.

## Top-Level Sections

- `inputs`
- `outputs`
- `image_scaling`
- `crop`
- `checks`
- `source_saturation_cut`
- `logging`
- `performance`
- `epsf`
- `patches`
- `patch_inputs`
- `patch_run`
- `moffat_psf`
- `merge`

## Operationally Important Options

### `crop.*`

- `enabled`: enable crop filtering + cropped image products
- `margin`: crop margin in pixels
- `display_downsample`: pre-crop white diagnostic DS
- `post_crop_display_downsample`: post-crop white diagnostic DS
- `overlay_catalog`: enable white+catalog overlay plot
- `overlay_downsample_full`: DS for overlay rendering

### `source_saturation_cut.*`

- `enabled`: remove sources near saturation before patch generation
- `radius_pix`: radial neighborhood used for saturation test
- `require_all_bands`: any-band vs all-bands removal criterion

### `performance.*`

- `frame_prep_workers`: workers for per-frame load/prep (`auto` or int)
- `white_stack_workers`: workers for white stack chunk accumulation (`auto` or int)

`auto` means capped by both CPU count and available jobs.

### `epsf.*`

- `epsf_ngrid`: ePSF grid size
- `ngrid`: patch grid subdivision per ePSF cell
- `parallel_bands`: build ePSF bands concurrently
- `max_workers`: worker count for ePSF band execution
- `skip_empty_epsf_patches`: skip ePSF cells with no input sources

### `patch_inputs.*`

- `skip_empty_patch`: skip writing payloads for patches with zero sources
- `max_workers`: async write worker count
- `gzip_compresslevel`: payload compression level

### `patch_run.*`

- `resume`: skip already completed patch outputs
- `max_workers`: subprocess concurrency
- `threads_per_process`: BLAS/OpenMP threads per subprocess
- `gal_model`: fallback source model when `TYPE` is missing
- `psf_model` / `psf_fallback_model`: PSF path choices
- `min_epsf_nstars_for_use`: low-star ePSF quality gate
- cutout controls (`cutout_*`, `no_cutouts`, `no_patch_overview`)

### `logging.*`

- `level`, `file`, `format`, `ignore_warnings`

Pipeline commands honor this section after config load.

## Compatibility Note

Defaults in `config.py` are kept aligned with `sample_config.yaml`. If they diverge, generated configs and implicit defaults may behave differently than expected.
