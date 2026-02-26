from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "inputs": {
        "tile_name": "your_tile_name",
        "input_catalog": "path/to/input_catalog.csv",
        "image_list_file": "path/to/image_list.txt",
        "gaiaxp_synphot_csv": "path/to/gaiaxp_synphot.csv",
    },
    "outputs": {
        "work_dir": "path/to/work_dir",
        "epsf_dir": "EPSFs",
        "patches_dir": "patches",
        "patch_inputs_dir": "patch_payloads",
        "tractor_out_dir": "outputs",
        "final_catalog": "output_catalog.csv",
        "cropped_images_dir": "cropped_images",
    },
    "image_scaling": {
        "zp_ref": 25.0,
    },
    "crop": {
        "enabled": True,
        "margin": 500,
        "plot_pre_crop": True,
        "display_downsample": 4,
        "post_crop_display_downsample": 4,
        "overlay_catalog": True,
        "overlay_downsample_full": 4,
        "overlay_downsample_crop": 4,
        "overlay_crop_box": [4000, 5000, 4000, 5000],
        "overlay_crop_enabled": True,
    },
    "checks": {
        "require_wcs_alignment": True,
        "wcs_tolerance": {
            "crval": 1e-6,
            "crpix": 1e-6,
            "cd": 1e-9,
            "cdelt": 1e-9,
        },
        "require_same_shape": True,
    },
    "source_saturation_cut": {
        "enabled": True,
        "radius_pix": 20.0,
        "require_all_bands": False,
    },
    "logging": {
        "ignore_warnings": True,
        "level": "INFO",
        "file": "auto",
        "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    },
    "plotting": {
        "dpi": 300,
    },
    "performance": {
        "frame_prep_workers": "auto",
        "white_stack_workers": "auto",
    },
    "epsf": {
        "use_gaiaxp": True,
        "epsf_ngrid": 7,
        "ngrid": 6,
        "skip_empty_epsf_patches": True,
        "thresh_sigma": 10.0,
        "minarea": 25,
        "cutout_size": 61,
        "edge_pad": 10,
        "min_sep": 30.0,
        "max_stars": 30,
        "q_range": [0.7, 1.3],
        "psfstar_mode": "sep+gaia",
        "gaia_mag_min": 10.0,
        "gaia_mag_max": 30.0,
        "gaia_snap_to_sep": True,
        "gaia_forced_k_fwhm": 1.5,
        "gaia_forced_rmin_pix": 3.0,
        "gaia_forced_rmax_pix": 15.0,
        "gaia_snr_min": 20.0,
        "gaia_match_r_pix": 5.0,
        "gaia_seed_sat_r": 5,
        "gaia_seed_sat_div": 1.3,
        "oversamp": 1,
        "maxiters": 30,
        "do_clip": True,
        "recenter_boxsize": 9,
        "final_psf_size": 55,
        "save_star_montage_max": 100,
        "background_subtract_patch": True,
        "background_boxsize": 96,
        "background_filtersize": 9,
        "background_source_mask_sigma": 3.0,
        "background_mask_dilate": 1,
        "star_local_bkg_subtract": True,
        "star_local_bkg_annulus_rin_pix": 20.0,
        "star_local_bkg_annulus_rout_pix": 30.0,
        "star_local_bkg_sigma": 3.0,
        "star_local_bkg_minpix": 30,
        "save_patch_background_diagnostics": True,
        "save_star_local_background_diagnostics": True,
        "diagnostics_show_colorbar": True,
        "save_growth_curve": True,
        "save_residual_diagnostics": True,
        "residual_diag_max_stars": 100,
        "parallel_bands": True,
        "max_workers": "auto",
    },
    "patches": {
        "halo_pix_min": 20,
    },
    "patch_inputs": {
        "save_patch_inputs": True,
        "skip_empty_patch": True,
        "include_wcs_in_payload": False,
        "max_workers": 16,
        "gzip_compresslevel": 1,
        "keep_payloads_in_memory": False,
    },
    "patch_run": {
        "enable_multi_band_simultaneous_fitting": True,
        "python_exe": "python",
        "resume": False,
        "max_workers": 32,
        "threads_per_process": 1,
        "gal_model": "exp",
        "n_opt_iters": 20,
        "dlnp_stop": 1e-6,
        "r_ap": 5.0,
        "eps_flux": 1e-4,
        "sersic_n_init": 3.0,
        "re_fallback_pix": 3.0,
        "psf_model": "pixelized",
        "psf_fallback_model": "moffat",
        "min_epsf_nstars_for_use": 2,
        "ncircular_gaussian_n": 3,
        "hybrid_psf_n": 9,
        "gaussian_mixture_n": 4,
        "fallback_default_fwhm_pix": 4.0,
        "fallback_stamp_radius_pix": 25.0,
        "cutout_size": 100,
        "cutout_max_sources": None,
        "cutout_start": 0,
        "cutout_data_p_lo": 1,
        "cutout_data_p_hi": 99,
        "cutout_model_p_lo": 1,
        "cutout_model_p_hi": 99,
        "cutout_resid_p_abs": 99,
        "no_cutouts": False,
        "no_patch_overview": False,
        "flag_maxiter_margin": 0,
    },
    "moffat_psf": {
        "fwhm_default_pix": 4.0,
        "beta": 3.5,
        "radius_pix": 25.0,
        "module_path": None,
    },
    "merge": {
        "pattern": "*_cat_fit.csv",
        "wcs_fits": None,
    },
    "zp": {
        "enabled": True,
        "inject_gaia_sources": True,
        "gaia_mag_min": 8.0,
        "gaia_mag_max": 18.0,
        "match_radius_arcsec": 3.0,
        "min_box_size_pix": 3000,
        "clip_sigma": 3.0,
        "clip_max_iters": 10,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _validate_user_config(user_cfg: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    if not isinstance(user_cfg, dict):
        raise TypeError("Config root must be a mapping (YAML dict).")
    for k, v in user_cfg.items():
        if k not in defaults:
            raise KeyError(f"Unknown config key: {_join_path(prefix, str(k))}")
        d = defaults[k]
        if isinstance(v, dict) and isinstance(d, dict):
            _validate_user_config(v, d, _join_path(prefix, str(k)))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_config_types(cfg: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    type_overrides: dict[str, tuple[type, ...]] = {
        "epsf.max_workers": (int, str),
        "performance.frame_prep_workers": (int, str),
        "performance.white_stack_workers": (int, str),
        "patch_run.cutout_max_sources": (int, type(None)),
        "moffat_psf.module_path": (str, Path, type(None)),
        "merge.wcs_fits": (str, Path, type(None)),
        "logging.file": (str, Path, type(None)),
    }
    for k, d in defaults.items():
        path = _join_path(prefix, str(k))
        if k not in cfg:
            continue
        v = cfg[k]
        if path in type_overrides:
            if not isinstance(v, type_overrides[path]):
                allowed = ", ".join(t.__name__ for t in type_overrides[path])
                raise TypeError(f"Config key '{path}' must be one of: {allowed}")
            continue
        if isinstance(d, dict):
            if not isinstance(v, dict):
                raise TypeError(f"Config key '{path}' must be a mapping (YAML dict).")
            _validate_config_types(v, d, path)
            continue
        if isinstance(d, list):
            if not isinstance(v, list):
                raise TypeError(f"Config key '{path}' must be a list.")
            continue
        if d is None:
            continue
        if isinstance(d, bool):
            if not isinstance(v, bool):
                raise TypeError(f"Config key '{path}' must be a bool.")
            continue
        if isinstance(d, int):
            if not _is_int(v):
                raise TypeError(f"Config key '{path}' must be an int.")
            continue
        if isinstance(d, float):
            if not (_is_int(v) or isinstance(v, float)):
                raise TypeError(f"Config key '{path}' must be a float.")
            continue
        if isinstance(d, str):
            if not isinstance(v, str):
                raise TypeError(f"Config key '{path}' must be a string.")
            continue


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg_dir = cfg_path.parent

    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    _validate_user_config(user_cfg, DEFAULT_CONFIG)
    cfg = _deep_update(DEFAULT_CONFIG, user_cfg)
    _validate_config_types(cfg, DEFAULT_CONFIG)

    cfg["config_path"] = cfg_path
    cfg["config_dir"] = cfg_dir

    inputs = cfg["inputs"]
    outputs = cfg["outputs"]

    inputs["input_catalog"] = _resolve_path(inputs["input_catalog"], cfg_dir)
    inputs["image_list_file"] = _resolve_path(inputs["image_list_file"], cfg_dir)
    inputs["gaiaxp_synphot_csv"] = _resolve_path(inputs["gaiaxp_synphot_csv"], cfg_dir)

    work_dir = _resolve_path(outputs["work_dir"], cfg_dir)
    outputs["work_dir"] = work_dir
    outputs["epsf_dir"] = _resolve_path(outputs["epsf_dir"], work_dir)
    outputs["patches_dir"] = _resolve_path(outputs["patches_dir"], work_dir)
    outputs["patch_inputs_dir"] = _resolve_path(outputs["patch_inputs_dir"], work_dir)
    outputs["tractor_out_dir"] = _resolve_path(outputs["tractor_out_dir"], work_dir)
    outputs["final_catalog"] = _resolve_path(outputs["final_catalog"], work_dir)
    outputs["cropped_images_dir"] = _resolve_path(outputs["cropped_images_dir"], work_dir)

    merge_cfg = cfg["merge"]
    if merge_cfg.get("wcs_fits"):
        merge_cfg["wcs_fits"] = _resolve_path(merge_cfg["wcs_fits"], cfg_dir)

    moffat_cfg = cfg["moffat_psf"]
    if moffat_cfg.get("module_path"):
        moffat_cfg["module_path"] = _resolve_path(moffat_cfg["module_path"], cfg_dir)

    logging_cfg = cfg["logging"]
    log_file = logging_cfg.get("file")
    if log_file is not None and str(log_file).strip().lower() != "auto":
        logging_cfg["file"] = _resolve_path(log_file, work_dir)

    return cfg


def write_sample_config(out_path: str | Path, overwrite: bool = False) -> Path:
    out = Path(out_path).expanduser()
    out_abs = out.resolve()
    sample_path = Path(__file__).resolve().parent / "data" / "sample_config.yaml"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample config not found: {sample_path}")
    if out_abs.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_abs}")
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    out_abs.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_abs
