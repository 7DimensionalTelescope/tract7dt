# Configuration

The pipeline is controlled by one YAML file. All pipeline commands require `--config /path/to/config.yaml`.

- Start from `tract7dt/data/sample_config.yaml` (or generate it with `tract7dt dump-config`).
- **Unknown keys are rejected** — misspelled keys produce a `KeyError` listing the offending key path.
- **Type validation** — each key's type is validated against the expected type. For example, passing a string where an integer is expected raises `TypeError`.
- **Missing keys** fall back to defaults defined in `tract7dt/config.py`.

## Path Resolution Rules

Understanding how relative paths are resolved is critical:

| Config section | Base directory for relative paths |
|---------------|----------------------------------|
| `inputs.input_catalog` | YAML config file directory |
| `inputs.image_list_file` | YAML config file directory |
| `inputs.gaiaxp_synphot_csv` | YAML config file directory |
| `outputs.work_dir` | YAML config file directory |
| `outputs.*` (all others) | `outputs.work_dir` |
| `merge.wcs_fits` | YAML config file directory |
| `moffat_psf.module_path` | YAML config file directory |
| `logging.file` | YAML config file directory |

Absolute paths are always used as-is. All relative paths are expanded with `Path.expanduser()` and resolved to absolute paths during config loading.

## Top-Level Sections

The YAML file is organized into these sections:

| Section | Purpose |
|---------|---------|
| [`inputs`](#inputs) | Paths to input data (catalog, images, GaiaXP) |
| [`outputs`](#outputs) | Paths to output directories and files |
| [`image_scaling`](#image_scaling) | Photometric scaling parameters |
| [`crop`](#crop) | Edge-crop filtering and diagnostic plots |
| [`checks`](#checks) | WCS and shape validation controls |
| [`source_saturation_cut`](#source_saturation_cut) | Pre-fit saturation-based source flagging |
| [`logging`](#logging) | Python logging configuration |
| [`performance`](#performance) | Parallelism controls for the load stage |
| [`epsf`](#epsf) | Effective PSF construction parameters |
| [`patches`](#patches) | Patch geometry |
| [`patch_inputs`](#patch_inputs) | Patch payload building |
| [`patch_run`](#patch_run) | Tractor fitting subprocess parameters |
| [`moffat_psf`](#moffat_psf) | Moffat PSF fallback parameters |
| [`merge`](#merge) | Final catalog merge settings |
| [`plotting`](#plotting) | Plot DPI settings |
| [`zp`](#zp) | Zero-point calibration using GaiaXP |

---

## `inputs`

Paths to the three input data files. See [Input Catalog Reference](input-catalog.md) for detailed format documentation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tile_name` | string | `"your_tile_name"` | Informational tile/field name. Does not affect pipeline behavior. |
| `input_catalog` | path | `"path/to/input_catalog.csv"` | Path to input source catalog CSV. **Must be set to your file.** |
| `image_list_file` | path | `"path/to/image_list.txt"` | Path to text file listing FITS image paths. **Must be set to your file.** |
| `gaiaxp_synphot_csv` | path | `"path/to/gaiaxp_synphot.csv"` | Path to GaiaXP synthetic photometry CSV. Set to `null` or omit if not using GaiaXP. |

## `outputs`

Paths to output directories and files. All paths except `work_dir` are resolved relative to `work_dir` if given as relative paths.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `work_dir` | path | `"path/to/work_dir"` | Base output directory. **Must be set to your directory.** Created if it does not exist. |
| `epsf_dir` | path | `"EPSFs"` | Directory for ePSF products (`.npy`, montages, meta). |
| `patches_dir` | path | `"patches"` | Directory for patch definition files (`patches.csv`, `patches.json`). |
| `patch_inputs_dir` | path | `"patch_payloads"` | Directory for compressed patch payloads (`*.pkl.gz`). |
| `tractor_out_dir` | path | `"outputs"` | Directory for per-patch Tractor fit outputs. |
| `final_catalog` | path | `"output_catalog.csv"` | Path for the merged final catalog CSV. |
| `cropped_images_dir` | path | `"cropped_images"` | Directory for white-stack and overlay diagnostic PNGs. |

## `image_scaling`

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `zp_ref` | float | `25.0` | mag (AB) | Reference zero-point. All images are scaled to this ZP. The scaling factor per image is `10^(-0.4 * (ZP_AUTO - zp_ref))`. |

## `crop`

Controls the removal of noisy edge regions (common in stacked images) and associated diagnostic plots.

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `enabled` | bool | `true` | — | Enable edge-crop filtering. When true, a margin is removed from all edges of the white stack and per-band arrays. Sources outside the crop box are **flagged** with `excluded_crop = True` and excluded from fitting. |
| `margin` | int | `500` | pixels | Width of the edge strip to remove on each side. The crop box is `[margin, W-margin) x [margin, H-margin)`. |
| `plot_pre_crop` | bool | `true` | — | Save a pre-crop white-stack diagnostic PNG with crop box overlay. |
| `display_downsample` | int | `4` | factor | Downsampling factor for the pre-crop white diagnostic plot (reduces rendering time and file size). |
| `post_crop_display_downsample` | int | `4` | factor | Downsampling factor for the post-crop white diagnostic plot. |
| `overlay_catalog` | bool | `true` | — | Render a white-stack + source overlay plot color-coded by TYPE. |
| `overlay_downsample_full` | int | `4` | factor | Downsampling factor for the full-field overlay plot. |
| `overlay_downsample_crop` | int | `4` | factor | Reserved for future zoomed overlay rendering. |
| `overlay_crop_box` | list of 4 int | `[4000, 5000, 4000, 5000]` | pixels `[x0, x1, y0, y1]` | Reserved zoom region for overlay. |
| `overlay_crop_enabled` | bool | `true` | — | Reserved. |

## `checks`

Validation gates applied during `load_inputs()` to catch incompatible images early.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `require_wcs_alignment` | bool | `true` | Enforce that all images share aligned WCS within tolerance. |
| `wcs_tolerance.crval` | float | `1e-6` | Maximum allowed absolute difference in CRVAL (degrees). |
| `wcs_tolerance.crpix` | float | `1e-6` | Maximum allowed absolute difference in CRPIX (pixels). |
| `wcs_tolerance.cd` | float | `1e-9` | Maximum allowed absolute difference in CD or PC matrix elements. |
| `wcs_tolerance.cdelt` | float | `1e-9` | Maximum allowed absolute difference in CDELT (degrees/pixel). |
| `require_same_shape` | bool | `true` | Enforce that all images have identical `(NAXIS2, NAXIS1)` dimensions. |

## `source_saturation_cut`

Pre-fit filter that **flags** sources located near saturated pixels. Flagged sources remain in the catalog (with `excluded_saturation = True`) but are excluded from patch assignment and fitting. This prevents the optimizer from wasting time on sources whose photometry is corrupted by detector saturation, while preserving all sources in the final output for completeness.

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `enabled` | bool | `true` | — | Enable saturation-based source flagging. |
| `radius_pix` | float | `20.0` | pixels | Search radius around each source position. If any pixel within this disk is flagged as saturated, the source is flagged as excluded. |
| `require_all_bands` | bool | `false` | — | `false`: flag if saturated in **any** band (conservative). `true`: flag only if saturated in **all** bands. |

## `logging`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ignore_warnings` | bool | `true` | Suppress Python `warnings` module output. |
| `level` | string | `"INFO"` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `file` | string, path, or null | `"auto"` | Log file destination. `"auto"`: write to `{work_dir}/{command}.log` (e.g. `run.log`, `merge.log`, `run-epsf.log`). Explicit path: resolved relative to `work_dir`. `null`: console only, no log file. Log files use append mode so that repeated runs accumulate in the same file. |
| `format` | string | `"%(asctime)s \| %(name)s \| %(levelname)s \| %(message)s"` | Python logging format string. |

## `performance`

Worker counts for the two parallelizable steps in `load_inputs()`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `frame_prep_workers` | int or `"auto"` | `"auto"` | Number of threads for parallel per-frame FITS loading and preparation. `"auto"` = `min(image_count, cpu_count)`. |
| `white_stack_workers` | int or `"auto"` | `"auto"` | Number of threads for parallel white-stack row-chunk accumulation. `"auto"` = `min(chunk_count, cpu_count)`. |

## `epsf`

Controls the construction of the effective PSF (ePSF) for each band in each spatial grid cell.

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `use_gaiaxp` | bool | `true` | — | Enable GaiaXP synphot catalog for PSF star seed selection. |
| `epsf_ngrid` | int | `7` | — | ePSF grid dimension. The image is divided into `epsf_ngrid x epsf_ngrid` cells, each getting its own ePSF. |
| `ngrid` | int | `6` | — | Patch subdivision within each ePSF cell. Each cell is further divided into `ngrid x ngrid` patches for fitting. Total patches (before pruning) = `epsf_ngrid^2 * ngrid^2`. |
| `skip_empty_epsf_patches` | bool | `true` | — | Skip ePSF cells that contain no input-catalog sources. Also restricts downstream patch definitions to active cells. |
| `thresh_sigma` | float | `10.0` | sigma | SEP source detection threshold. |
| `minarea` | int | `25` | pixels | Minimum contiguous pixel area for SEP source detection. |
| `cutout_size` | int | `61` | pixels | Size of the cutout box extracted around each PSF star candidate for ePSF building. |
| `edge_pad` | int | `10` | pixels | Edge padding: stars within this many pixels of the cutout edge are rejected. |
| `min_sep` | float | `30.0` | pixels | Minimum separation between accepted PSF stars (prevents blended pairs). |
| `max_stars` | int | `30` | — | Maximum number of PSF stars per ePSF cell. |
| `q_range` | list [float, float] | `[0.7, 1.3]` | dimensionless | Allowed range of SEP roundness parameter `a/b` for PSF star candidates. Values near 1.0 select round sources. |
| `psfstar_mode` | string | `"sep+gaia"` | — | PSF star selection strategy: `"sep"` (SEP only), `"gaia"` (Gaia only), `"sep+gaia"` (Gaia seeds + SEP supplement). |
| `gaia_mag_min` | float | `10.0` | mag (AB) | Minimum GaiaXP magnitude for PSF star seeds (rejects saturated stars). |
| `gaia_mag_max` | float | `30.0` | mag (AB) | Maximum GaiaXP magnitude for PSF star seeds (rejects faint stars). |
| `gaia_snap_to_sep` | bool | `true` | — | Refine GaiaXP seed positions by snapping to nearest SEP detection. |
| `gaia_forced_k_fwhm` | float | `1.5` | dimensionless | Scale factor: forced aperture radius = `k_fwhm * FWHM_pix`. |
| `gaia_forced_rmin_pix` | float | `3.0` | pixels | Minimum forced aperture radius. |
| `gaia_forced_rmax_pix` | float | `15.0` | pixels | Maximum forced aperture radius. |
| `gaia_snr_min` | float | `20.0` | dimensionless | Minimum signal-to-noise ratio for PSF star candidates (forced aperture). |
| `gaia_match_r_pix` | float | `5.0` | pixels | Maximum distance for Gaia-to-SEP position matching (snap radius). |
| `gaia_seed_sat_r` | int | `5` | pixels | Box half-size for local saturation check around each seed position. |
| `gaia_seed_sat_div` | float | `1.3` | dimensionless | Saturation threshold divisor: local max must be below `SATURATE / gaia_seed_sat_div`. |
| `oversamp` | int | `1` | factor | ePSF oversampling factor. |
| `maxiters` | int | `30` | — | Maximum EPSFBuilder iterations. |
| `do_clip` | bool | `true` | — | Enable sigma-clipping during ePSF building. |
| `recenter_boxsize` | int | `9` | pixels | Recentering box size for EPSFBuilder. |
| `final_psf_size` | int | `55` | pixels | Final ePSF stamp size (center-cropped from the built ePSF). |
| `save_star_montage_max` | int | `100` | — | Maximum number of stars shown in diagnostic star montage PNGs. |
| `background_subtract_patch` | bool | `true` | — | Subtract a local 2D background map (SEP) per ePSF cell before star selection and ePSF building. |
| `background_boxsize` | int | `96` | pixels | SEP background mesh size (`bw=bh`) used for patch-level background estimation. |
| `background_filtersize` | int | `9` | — | SEP background smoothing filter size (`fw=fh`). |
| `background_source_mask_sigma` | float | `3.0` | sigma | Additional source mask threshold for background fitting: pixels above `median + N*sigma_sky` are masked. |
| `background_mask_dilate` | int | `1` | iterations | Dilation iterations applied to the source mask before background fitting. |
| `star_local_bkg_subtract` | bool | `true` | — | Subtract local annulus background per extracted star cutout before EPSFBuilder. |
| `star_local_bkg_annulus_rin_pix` | float | `20.0` | pixels | Inner radius of local annulus for per-star background estimation. |
| `star_local_bkg_annulus_rout_pix` | float | `30.0` | pixels | Outer radius of local annulus for per-star background estimation. |
| `star_local_bkg_sigma` | float | `3.0` | sigma | Sigma-clipping threshold for annulus background estimate. |
| `star_local_bkg_minpix` | int | `30` | pixels | Minimum valid annulus pixels required to use local background; else fallback to 0. |
| `save_patch_background_diagnostics` | bool | `true` | — | Save `background_diagnostics.png` (raw/background/bg-sub panels) for each ePSF cell. Disable to reduce I/O and plotting overhead. |
| `save_star_local_background_diagnostics` | bool | `true` | — | Save `star_local_background_diagnostics.png` (per-star raw/annulus/bg-sub grid). Disable for major speed/memory savings when many stars are used. |
| `diagnostics_show_colorbar` | bool | `true` | — | Show colorbars in ePSF background diagnostics (`background_diagnostics.png`, `star_local_background_diagnostics.png`). |
| `save_growth_curve` | bool | `true` | — | Save `epsf_growth_curve.png` and write EE/edge metrics into per-patch `meta.json`. |
| `save_residual_diagnostics` | bool | `true` | — | Save `epsf_residual_diagnostics.png` from fitted-star residual stacks. |
| `residual_diag_max_stars` | int | `100` | — | Max number of fitted stars used in residual diagnostics. |
| `parallel_bands` | bool | `true` | — | Build ePSF for different bands concurrently. |
| `max_workers` | int or `"auto"` | `"auto"` | — | Worker count for parallel band execution. `"auto"` = `min(n_bands, cpu_count)`. |

## `patches`

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `halo_pix_min` | int | `20` | pixels | Minimum halo size around each patch. The actual halo is `max((final_psf_size - 1) / 2 + 2, halo_pix_min)`. The halo ensures that source models near patch boundaries are properly evaluated. Increase this if your targets have large angular extent. |

## `patch_inputs`

Controls how per-patch data payloads are built and serialized.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_patch_inputs` | bool | `true` | Write `*.pkl.gz` payload files to disk. Must be `true` for CLI-based patch runs. |
| `skip_empty_patch` | bool | `true` | Skip writing payloads for patches containing zero sources (saves I/O and disk). |
| `include_wcs_in_payload` | bool | `false` | Store WCS objects in patch payloads. Reserved for future use. |
| `max_workers` | int | `16` | Thread worker count for parallel payload writing. Reduce if I/O-bound. |
| `gzip_compresslevel` | int | `1` | gzip compression level for payloads (1=fast, 9=best compression). |
| `keep_payloads_in_memory` | bool | `false` | Keep built payloads in RAM for potential in-process use. |

## `patch_run`

Controls the Tractor fitting subprocesses that run on each patch.

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `enable_multi_band_simultaneous_fitting` | bool | `true` | — | **Fitting mode.** `true`: fit all bands simultaneously (position and morphology shared across bands, per-band fluxes). `false`: fit each band independently (position, morphology, and flux all vary per band). See [Pipeline Behavior — Stage 5](pipeline-behavior.md#stage-5-run-patch-subprocesses) for details. |
| `python_exe` | string | `"python"` | — | Python executable for launching subprocesses. |
| `resume` | bool | `false` | — | Skip patches whose output directories already exist. **Caution:** if you change config parameters and re-run with `resume: true`, previously completed patches will NOT be re-fit with the new parameters. |
| `max_workers` | int | `32` | — | Maximum concurrent subprocess count. |
| `threads_per_process` | int | `1` | — | BLAS/OpenMP thread count per subprocess. Avoid oversubscription: `max_workers * threads_per_process ≤ cpu_count`. |
| `gal_model` | string | `"exp"` | — | Fallback source model when `TYPE` is missing or unrecognized. Choices: `"dev"`, `"exp"`, `"sersic"`, `"star"`. |
| `n_opt_iters` | int | `20` | — | Maximum Tractor optimization iterations. In multi-band mode this is per patch; in single-band mode this is per band within each patch. |
| `dlnp_stop` | float | `1e-6` | dimensionless | Convergence criterion: stop when the change in log-probability is below this value. |
| `r_ap` | float | `5.0` | pixels | Aperture radius for SEP-based flux initialization. |
| `eps_flux` | float | `1e-4` | scaled counts | Minimum flux floor. All initial fluxes are clamped to at least this value. |
| `sersic_n_init` | float | `3.0` | dimensionless | Initial Sersic index for `SERSIC`-type sources. |
| `re_fallback_pix` | float | `3.0` | pixels | Default effective radius when `Re` column is absent or `NaN`. |
| `psf_model` | string | `"pixelized"` | — | PSF model when ePSF is available. Choices: `"pixelized"` (direct pixel array), `"hybrid"` (HybridPixelizedPSF), `"gaussian_mixture_from_stamp"` (fit Gaussian mixture to ePSF stamp). |
| `psf_fallback_model` | string | `"moffat"` | — | PSF model when ePSF is missing or low-quality. Choices: `"gaussian_mixture"`, `"ncircular_gaussian"`, `"moffat"`. |
| `min_epsf_nstars_for_use` | int | `2` | — | Quality gate: if an ePSF cell used fewer than this many stars, treat the ePSF as unreliable and use the fallback PSF model. |
| `ncircular_gaussian_n` | int | `3` | — | Number of components for `NCircularGaussianPSF` fallback. Allowed: 1, 2, 3. |
| `hybrid_psf_n` | int | `9` | — | N parameter for `HybridPixelizedPSF`. |
| `gaussian_mixture_n` | int | `4` | — | Number of Gaussian components for `GaussianMixturePSF.fromStamp()`. |
| `fallback_default_fwhm_pix` | float | `4.0` | pixels | Default FWHM when `PEEING`/`SEEING` is missing from the FITS header (used for fallback PSF). |
| `fallback_stamp_radius_pix` | float | `25.0` | pixels | Radius of the Gaussian stamp used to synthesize a fallback PSF. |
| `cutout_size` | int | `100` | pixels | Side length of per-source cutout montages. |
| `cutout_max_sources` | int or null | `null` | — | Limit the number of source cutout montages per patch. `null` = all sources. |
| `cutout_start` | int | `0` | — | Start index for cutout montage generation. |
| `cutout_data_p_lo` | float | `1` | percentile | Low percentile for data cutout display scaling. |
| `cutout_data_p_hi` | float | `99` | percentile | High percentile for data cutout display scaling. |
| `cutout_model_p_lo` | float | `1` | percentile | Low percentile for model cutout display scaling. |
| `cutout_model_p_hi` | float | `99` | percentile | High percentile for model cutout display scaling. |
| `cutout_resid_p_abs` | float | `99` | percentile | Absolute residual percentile for symmetric color limits. |
| `no_cutouts` | bool | `false` | — | Disable per-source cutout montage generation (saves disk and time). |
| `no_patch_overview` | bool | `false` | — | Disable per-patch overview plot (saves disk and time). |
| `flag_maxiter_margin` | int | `0` | — | Margin for the `opt_hit_max_iters` flag: set to true if the optimizer did not converge and used at least `n_opt_iters - margin` iterations. |

## `moffat_psf`

Parameters for the Moffat PSF model, used when `patch_run.psf_fallback_model: "moffat"`.

| Key | Type | Default | Unit | Description |
|-----|------|---------|------|-------------|
| `fwhm_default_pix` | float | `4.0` | pixels | Legacy key. Use `patch_run.fallback_default_fwhm_pix` instead. |
| `beta` | float | `3.5` | dimensionless | Moffat beta (power-law slope) parameter. Typical values: 2.5–4.5. |
| `radius_pix` | float | `25.0` | pixels | Moffat PSF stamp radius. |
| `module_path` | path or null | `null` | — | Legacy. |

## `merge`

Controls how per-patch fit results are joined back to the base catalog (augmented catalog when `zp.enabled`, otherwise original input catalog).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pattern` | string | `"*_cat_fit.csv"` | Glob pattern to find per-patch fit CSV files under `outputs.tractor_out_dir`. |
| `wcs_fits` | path or null | `null` | Optional FITS file whose WCS is used to compute sky coordinates from fitted pixel positions (`RA_fit`/`DEC_fit` in multi-band mode; `RA_{band}_fit`/`DEC_{band}_fit` in single-band mode). If null, uses the WCS from the first loaded image. |

## `plotting`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dpi` | int | `300` | DPI for all diagnostic PNG plots (white-stack overlays, ePSF stamps, patch overviews, source montages, ZP plots, augmentation overlay). |

## `zp`

Controls zero-point calibration using GaiaXP synthetic photometry. When enabled, Gaia synphot sources are matched (and optionally injected) into the input catalog before fitting, and per-band ZP is computed after merge.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable zero-point calibration. When true, two additional pipeline stages are activated: Gaia catalog augmentation (after load) and ZP computation (after merge). |
| `inject_gaia_sources` | bool | `true` | `true`: inject unmatched Gaia sources as new catalog rows for ZP calibration. `false`: only match existing input sources against Gaia (adding `gaia_source_id` and backfilling missing FLUX values) without injecting new rows. Use `false` when you already have reference sources in the input catalog. |
| `gaia_mag_min` | float | `8.0` | Minimum `phot_g_mean_mag` for Gaia source selection. |
| `gaia_mag_max` | float | `18.0` | Maximum `phot_g_mean_mag` for Gaia source selection. |
| `match_radius_arcsec` | float | `3.0` | RA/DEC match radius (arcsec) for deduplication against the input catalog. |
| `min_box_size_pix` | int | `3000` | Minimum side length (pixels) of the square Gaia selection bounding box. |
| `clip_sigma` | float | `3.0` | MAD-based sigma clipping threshold for ZP computation. |
| `clip_max_iters` | int | `10` | Maximum sigma-clipping iterations. |

!!! note "Re-running ZP independently"
    `tract7dt compute-zp --config ...` re-runs only the ZP computation step on the existing merged catalog. Only `clip_sigma` and `clip_max_iters` take effect in this mode. The augmentation parameters (`gaia_mag_min`, `gaia_mag_max`, `match_radius_arcsec`, `min_box_size_pix`) are baked into the catalog at augmentation time and require a full pipeline re-run to change.

## Compatibility Note

Defaults in `config.py` are kept aligned with `sample_config.yaml`. If they diverge, generated configs and implicit defaults may behave differently than expected.
