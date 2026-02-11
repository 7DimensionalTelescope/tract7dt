# Pipeline Behavior

This page describes runtime behavior as implemented in code.

## Stage 1: Load Inputs and Validate

`load_inputs()` performs:

1. Read catalog and image list.
2. Parallel per-frame preparation:
   - load FITS image/header
   - required header checks
   - scale image to reference zeropoint
   - build bad/saturation masks
   - build sky and total sigma arrays
   - construct WCS
3. Validate shape/WCS consistency across frames.
4. Build white image and white noise map (parallel row chunks).
5. Apply crop (if enabled):
   - filter sources outside crop
   - crop per-band arrays and WCS
6. Apply saturation-cut filter (optional).
7. Render diagnostics (pre-crop, post-crop, overlay).

The stage logs a timing summary:

- prep, white, crop, saturation, overlay, total.

## ePSF Cell Activity Filtering

If `epsf.skip_empty_epsf_patches=true`:

- active ePSF cells are computed from current input-catalog source positions.
- ePSF generation runs only on active cells.
- patch definitions are also restricted to active ePSF cells.

This reduces unnecessary work in sparse fields.

## Patch Input Filtering

`patch_inputs.skip_empty_patch` acts later:

- true: skip writing patch payloads with zero sources.
- false: write payloads even for empty patch subcells.

Combination behavior:

- `skip_empty_epsf_patches=true`, `skip_empty_patch=false`
  - active ePSF cells are pruned
  - empty subpatches inside active ePSF cells are still written/run.

## Source Exclusion Tracking

Sources filtered out before fitting are tracked and merged back into final output:

- `excluded_crop`
- `excluded_saturation`
- `excluded_any`
- `excluded_reason` (`crop`, `saturation`, `crop+saturation`, or empty)

Keying uses the same merge keys as final merge (`ID` preferred, otherwise `RA/DEC`).

## Overlay TYPE Behavior

Overlay plotting categories:

- `STAR` -> cyan markers
- `GAL/EXP/DEV/SERSIC` -> magenta markers
- unknown/missing/non-standard type -> yellow square markers labeled:
  - `UNKNOWN (N=...) (fallback=<patch_run.gal_model>)`

If `TYPE` column is missing entirely, all plotted sources are treated as unknown for overlay diagnostics.

## Patch Run Progress Behavior

Patch subprocess stage emits live progress:

- TTY: in-place progress line
- non-TTY: periodic heartbeat logs

It reports done/total, running count, fail count, rate, and ETA.

## ePSF Progress Behavior

In parallel band mode with TTY, ePSF progress renders one live line per worker slot.

- If workers < bands, slots are reused.
- Slot notices indicate handoff (`done <band> -> start <next>`).
