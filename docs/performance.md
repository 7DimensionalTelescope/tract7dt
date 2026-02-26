# Performance Tuning

This page focuses on speed controls that do not intentionally reduce scientific quality.

## High-Impact Options

### 1) Input load parallelism

- `performance.frame_prep_workers`
  - Controls per-frame FITS preparation concurrency.
  - `auto` = min(image_count, cpu_count).

### 2) White-stack parallelism

- `performance.white_stack_workers`
  - Controls row-chunk white stack accumulation concurrency.
  - `auto` = min(chunk_count, cpu_count).

### 3) ePSF sparsity pruning

- `epsf.skip_empty_epsf_patches: true`
  - skips ePSF cells with no source occupancy.
  - also reduces downstream patch definition scope.

### 4) Empty patch payload pruning

- `patch_inputs.skip_empty_patch: true`
  - avoids writing/running empty patch payloads.

## Useful Runtime Indicators

`load_inputs` timing summary identifies bottlenecks:

- `prep`: frame load + prep
- `white`: white-stack assembly
- `crop`: crop/filter/slice + post-crop plot
- `sat`: saturation filtering
- `overlay`: overlay plot
- `total`: overall load stage

## Plotting Cost Controls

If diagnostics are costly:

**Downsampling** (reduces plot resolution, not science):

- `crop.display_downsample` (pre-crop white)
- `crop.post_crop_display_downsample` (post-crop white)
- `crop.overlay_downsample_full` (overlay plot)

**ePSF diagnostic toggles** (disable to skip plot generation entirely):

- `epsf.save_star_local_background_diagnostics: false` — **highest impact**. Generates a multi-row plot for every used star in every ePSF cell. Disabling this provides major speed and I/O savings in dense fields.
- `epsf.save_patch_background_diagnostics: false` — skips per-cell background diagnostic panels.
- `epsf.save_growth_curve: false` — skips ePSF growth curve plots.
- `epsf.save_residual_diagnostics: false` — skips ePSF residual diagnostic plots.
- `epsf.diagnostics_show_colorbar: false` — removes colorbars from background diagnostics (minor savings).

**Patch fitting diagnostic toggles:**

- `patch_run.no_cutouts: true` — skips per-source cutout montages. Significant savings when many sources are fit.
- `patch_run.no_patch_overview: true` — skips per-patch overview plots.
- `patch_run.cutout_max_sources: N` — limits cutout montages to the first N sources per patch.

All diagnostic toggles affect rendering/I/O only, not fit model internals.

## Concurrency Caveats

- Higher workers can increase memory pressure.
- For subprocess patch runs, coordinate:
  - `patch_run.max_workers`
  - `patch_run.threads_per_process`
- Avoid thread oversubscription on shared systems.

## Practical Starting Point

```yaml
performance:
  frame_prep_workers: "auto"
  white_stack_workers: "auto"

epsf:
  skip_empty_epsf_patches: true

patch_inputs:
  skip_empty_patch: true
```

Then tune worker counts manually if system load or memory indicates constraints.
