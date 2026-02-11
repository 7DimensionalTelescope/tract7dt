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

- `crop.display_downsample` (pre-crop white)
- `crop.post_crop_display_downsample` (post-crop white)
- `crop.overlay_downsample_full` (overlay plot)

These affect diagnostic rendering workload, not fit model internals.

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
