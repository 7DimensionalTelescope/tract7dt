"""
Patch-level Tractor runner for tract7dt.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import sep
from astropy.io import fits

from tractor import ConstantSky, Fluxes, FluxesPhotoCal, Image, NullWCS, PixPos, PointSource, Tractor
from tractor.constrained_optimizer import ConstrainedOptimizer
from tractor.ellipses import EllipseESoft
from tractor.galaxy import DevGalaxy, ExpGalaxy
from tractor.psf import GaussianMixturePSF, HybridPixelizedPSF, NCircularGaussianPSF, PixelizedPSF
from tractor.sersic import SersicGalaxy, SersicIndex
from .moffat_psf import MoffatPSF

sep.set_sub_object_limit(200000)
sep.set_extract_pixstack(1000000)

_PATCH_PLOT_DPI = 150


@dataclass
class PatchRunConfig:
    # Fitting mode
    enable_multi_band_simultaneous_fitting: bool = True

    # Optimization
    n_opt_iters: int = 200
    dlnp_stop: float = 1e-6
    # Flag "non-converged" if we effectively hit max iters.
    # If not converged and niters >= (n_opt_iters - margin), set opt_hit_max_iters=True.
    flag_maxiter_margin: int = 5

    # Catalog init
    r_ap: float = 5.0
    eps_flux: float = 1e-4
    gal_model: str = "dev"  # dev|exp|sersic|star
    sersic_n_init: float = 3.0
    re_fallback_pix: float = 3.0

    # Output cutouts
    save_cutouts: bool = True
    cutout_size_pix: int = 100
    cutout_max_sources: Optional[int] = None
    cutout_start: int = 0
    cutout_data_p_lo: float = 1
    cutout_data_p_hi: float = 99
    cutout_model_p_lo: float = 1
    cutout_model_p_hi: float = 99
    cutout_resid_p_abs: float = 99

    # Per-patch overview
    save_patch_overview: bool = True

    # PSF choices:
    # psf_model: used when ePSF exists
    # psf_fallback_model: used when ePSF is missing
    psf_model: str = "hybrid"  # hybrid|pixelized|gaussian_mixture_from_stamp
    psf_fallback_model: str = "gaussian_mixture"  # gaussian_mixture|ncircular_gaussian|moffat
    min_epsf_nstars_for_use: int = 5
    ncircular_gaussian_n: int = 3  # allowed: 1|2|3
    hybrid_psf_n: int = 4
    gaussian_mixture_n: int = 4
    fallback_default_fwhm_pix: float = 4.0
    fallback_stamp_radius_pix: float = 25.0
    moffat_beta: float = 3.5
    moffat_radius_pix: float = 25.0

    # Plotting
    plot_dpi: int = 150


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _make_logger(outdir: Path, tag: str) -> logging.Logger:
    _ensure_dir(outdir)
    log = logging.getLogger(f"tractor_patch.{tag}")
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(outdir / f"{tag}.log", mode="w")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        log.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        log.addHandler(sh)
    return log


def _patch_tag_from_meta(meta: dict | None) -> str:
    if not meta:
        return "patch"
    t = meta.get("patch_tag", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "patch"


def _roi_bbox_from_meta(meta: dict) -> tuple[int, int, int, int]:
    if "roi_bbox" in meta:
        b = meta["roi_bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    keys = ("x0_roi", "x1_roi", "y0_roi", "y1_roi")
    if all(k in meta for k in keys):
        return (int(meta["x0_roi"]), int(meta["x1_roi"]), int(meta["y0_roi"]), int(meta["y1_roi"]))
    raise ValueError("meta must contain roi_bbox or (x0_roi,x1_roi,y0_roi,y1_roi)")


def load_patch_inputs(pkl_gz_path: str | Path) -> dict:
    p = Path(pkl_gz_path)
    with gzip.open(p, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Patch input payload must be a dict")
    return payload


def _fallback_fwhm_from_header(
    fp: str,
    *,
    default_fwhm_pix: float,
    log: logging.Logger | None = None,
) -> float:
    fwhm_pix = float(default_fwhm_pix)
    used_default = True
    try:
        with fits.open(fp, memmap=True) as hdul:
            hdr = hdul[0].header
        v = hdr.get("PEEING", None)
        if v is None:
            v = hdr.get("SEEING", None)
        if v is not None and np.isfinite(v) and v > 0:
            fwhm_pix = float(v)
            used_default = False
    except Exception:
        pass
    if used_default and log is not None:
        log.info("PEEING/SEEING missing/invalid in %s; using default_fwhm_pix=%s", fp, str(default_fwhm_pix))
    return max(float(fwhm_pix), 1.0)


def _gaussian_stamp_from_fwhm(*, fwhm_pix: float, radius_pix: float) -> np.ndarray:
    sigma = max(float(fwhm_pix) / 2.3548200450309493, 1e-3)
    r = max(1, int(np.ceil(float(radius_pix))))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    rr2 = (xx * xx + yy * yy).astype(np.float64)
    img = np.exp(-0.5 * rr2 / (sigma * sigma))
    s = float(np.nansum(img))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Failed to build Gaussian fallback stamp.")
    return (img / s).astype(np.float32)


def _center_crop_or_pad(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=float)
    y_in, x_in = arr.shape
    y_out, x_out = shape
    y0_in = max(0, (y_in - y_out) // 2)
    x0_in = max(0, (x_in - x_out) // 2)
    y0_out = max(0, (y_out - y_in) // 2)
    x0_out = max(0, (x_out - x_in) // 2)
    y1_in = y0_in + min(y_in, y_out)
    x1_in = x0_in + min(x_in, x_out)
    y1_out = y0_out + min(y_in, y_out)
    x1_out = x0_out + min(x_in, x_out)
    out[y0_out:y1_out, x0_out:x1_out] = arr[y0_in:y1_in, x0_in:x1_in]
    return out


def _psf_to_array(psf: Any, shape: tuple[int, int]) -> np.ndarray | None:
    for kwargs in ({}, {"stamp": int(shape[0])}, {"size": int(shape[0])}, {"halfsize": int(shape[0] // 2)}):
        try:
            patch = psf.getPointSourcePatch(0.0, 0.0, **kwargs)
            if patch is None:
                continue
            if hasattr(patch, "getImage"):
                arr = np.asarray(patch.getImage(), dtype=float)
            elif hasattr(patch, "image"):
                arr = np.asarray(patch.image, dtype=float)
            else:
                arr = np.asarray(patch, dtype=float)
            if arr.size:
                return _center_crop_or_pad(arr, shape)
        except Exception:
            continue
    if hasattr(psf, "getImage"):
        try:
            arr = np.asarray(psf.getImage(), dtype=float)
            if arr.size:
                return _center_crop_or_pad(arr, shape)
        except Exception:
            pass
    if hasattr(psf, "pix"):
        try:
            arr = np.asarray(psf.pix.getImage(), dtype=float)
            if arr.size:
                return _center_crop_or_pad(arr, shape)
        except Exception:
            pass
    return None


def _save_hybrid_psf_diagnostics(
    *,
    epsf_arr: np.ndarray,
    hybrid_psf: HybridPixelizedPSF,
    outpath: Path,
    log: logging.Logger,
) -> None:
    gauss_arr = _psf_to_array(hybrid_psf.gauss, epsf_arr.shape)
    if gauss_arr is not None:
        gauss_arr = np.nan_to_num(gauss_arr, nan=0.0, posinf=0.0, neginf=0.0)
        gs = float(np.sum(gauss_arr))
        if gs > 0:
            gauss_arr = gauss_arr / gs

    gauss_resid = (epsf_arr - gauss_arr) if gauss_arr is not None else None

    dvmin, dvmax = _robust_limits_percentile(epsf_arr, p_lo=1, p_hi=99)
    gvmin, gvmax = _robust_limits_percentile(gauss_arr, p_lo=1, p_hi=99) if gauss_arr is not None else (None, None)
    grvmin, grvmax = (
        _robust_resid_limits_percentile(gauss_resid, p_abs=99)
        if gauss_resid is not None
        else (None, None)
    )

    if gauss_arr is None:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        panels = [
            ("ePSF (normalized)", epsf_arr, dvmin, dvmax, "ePSF"),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        panels = [
            ("ePSF (normalized)", epsf_arr, dvmin, dvmax, "ePSF"),
            ("Gaussian model (normalized)", gauss_arr, gvmin, gvmax, "Gauss"),
            ("ePSF - gauss", gauss_resid, grvmin, grvmax, "Gauss resid"),
        ]
    for ax, (title, arr, vmin, vmax, cblab) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        amin, amax = _true_minmax(arr)
        if amin is not None and amax is not None:
            sm = ScalarMappable(norm=Normalize(vmin=amin, vmax=amax), cmap=im.cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, extend="neither")
            cb.set_label(cblab)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=_PATCH_PLOT_DPI)
    plt.close(fig)
    log.info("[PSF] Wrote hybrid PSF diagnostics: %s", str(outpath))


def _save_gaussian_mixture_psf_diagnostics(
    *,
    epsf_arr: np.ndarray,
    gauss_psf: GaussianMixturePSF,
    outpath: Path,
    log: logging.Logger,
) -> None:
    gauss_arr = _psf_to_array(gauss_psf, epsf_arr.shape)
    if gauss_arr is None:
        raise RuntimeError("Could not render GaussianMixturePSF to array for diagnostics.")
    gauss_arr = np.nan_to_num(gauss_arr, nan=0.0, posinf=0.0, neginf=0.0)
    gs = float(np.sum(gauss_arr))
    if gs > 0:
        gauss_arr = gauss_arr / gs

    resid = epsf_arr - gauss_arr

    dvmin, dvmax = _robust_limits_percentile(epsf_arr, p_lo=1, p_hi=99)
    gvmin, gvmax = _robust_limits_percentile(gauss_arr, p_lo=1, p_hi=99)
    rvmin, rvmax = _robust_resid_limits_percentile(resid, p_abs=99)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("ePSF (normalized)", epsf_arr, dvmin, dvmax, "ePSF"),
        ("GaussianMixture model (normalized)", gauss_arr, gvmin, gvmax, "Gauss"),
        ("ePSF - gauss", resid, rvmin, rvmax, "Gauss resid"),
    ]
    for ax, (title, arr, vmin, vmax, cblab) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        amin, amax = _true_minmax(arr)
        if amin is not None and amax is not None:
            sm = ScalarMappable(norm=Normalize(vmin=amin, vmax=amax), cmap=im.cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, extend="neither")
            cb.set_label(cblab)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=_PATCH_PLOT_DPI)
    plt.close(fig)
    log.info("[PSF] Wrote GaussianMixture PSF diagnostics: %s", str(outpath))


def build_psf_by_band_for_patch(
    *,
    per_filter_patch: dict,
    epsf_root: Path,
    epsf_tag: str,
    cfg: PatchRunConfig,
    log: logging.Logger,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    psf_by_band: dict[str, Any] = {}
    psf_audit: dict[str, dict[str, Any]] = {}
    for band, d in per_filter_patch.items():
        band = str(band)
        epsf_path = epsf_root / band / str(epsf_tag) / "epsf.npy"
        band_audit: dict[str, Any] = dict(
            epsf_path=str(epsf_path),
            epsf_exists=bool(epsf_path.exists()),
            used_epsf=False,
            fallback_reason=None,
            epsf_nstars_used=None,
            min_epsf_nstars_for_use=int(max(0, int(getattr(cfg, "min_epsf_nstars_for_use", 5)))),
            psf_model=None,
        )
        use_epsf = epsf_path.exists()
        fallback_reason = "missing ePSF"
        if use_epsf:
            min_n = max(0, int(getattr(cfg, "min_epsf_nstars_for_use", 5)))
            if min_n > 0:
                meta_path = epsf_path.parent / "meta.json"
                nstars_used = None
                try:
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        v = meta.get("nstars_used", None)
                        if v is not None:
                            nstars_used = int(v)
                            band_audit["epsf_nstars_used"] = int(nstars_used)
                except Exception as e:
                    log.warning("[PSF] %s: failed to read ePSF meta %s: %s", band, str(meta_path), str(e))
                if nstars_used is not None and nstars_used < min_n:
                    use_epsf = False
                    fallback_reason = (
                        f"low-star ePSF (nstars_used={nstars_used} < min_epsf_nstars_for_use={min_n})"
                    )
                    log.warning(
                        "[PSF] %s: ePSF exists at %s but %s; using fallback path",
                        band,
                        str(epsf_path),
                        fallback_reason,
                    )

        if use_epsf:
            epsf_arr = np.load(epsf_path)
            epsf_arr = np.asarray(epsf_arr, dtype=np.float32)
            epsf_arr = np.nan_to_num(epsf_arr, nan=0.0, posinf=0.0, neginf=0.0)
            s = float(np.sum(epsf_arr))
            if s > 0:
                epsf_arr = epsf_arr / s
            pixpsf = PixelizedPSF(epsf_arr)
            psf_model = str(getattr(cfg, "psf_model", "hybrid")).lower()
            band_audit["used_epsf"] = True
            band_audit["psf_model"] = str(psf_model)
            if psf_model == "pixelized":
                psf_by_band[band] = pixpsf
            elif psf_model == "gaussian_mixture_from_stamp":
                gauss = GaussianMixturePSF.fromStamp(epsf_arr, N=max(1, int(cfg.gaussian_mixture_n)))
                psf_by_band[band] = gauss
                try:
                    _save_gaussian_mixture_psf_diagnostics(
                        epsf_arr=epsf_arr,
                        gauss_psf=gauss,
                        outpath=epsf_path.parent / "gaussian_mixture_psf_diagnostics.png",
                        log=log,
                    )
                except Exception as e:
                    log.warning("[PSF] GaussianMixture PSF diagnostics failed: %s", str(e))
            else:
                hybrid = HybridPixelizedPSF(pixpsf, gauss=None, N=int(cfg.hybrid_psf_n), cx=0.0, cy=0.0)
                psf_by_band[band] = hybrid
                try:
                    _save_hybrid_psf_diagnostics(
                        epsf_arr=epsf_arr,
                        hybrid_psf=hybrid,
                        outpath=epsf_path.parent / "hybrid_psf_diagnostics.png",
                        log=log,
                    )
                except Exception as e:
                    log.warning("[PSF] Hybrid PSF diagnostics failed: %s", str(e))
        else:
            fp = d.get("fp", None)
            if fp:
                fwhm_pix = _fallback_fwhm_from_header(
                    str(fp),
                    default_fwhm_pix=cfg.fallback_default_fwhm_pix,
                    log=log,
                )
            else:
                fwhm_pix = float(cfg.fallback_default_fwhm_pix)
            psf_fallback_model = str(getattr(cfg, "psf_fallback_model", "gaussian_mixture")).lower()
            band_audit["used_epsf"] = False
            band_audit["fallback_reason"] = str(fallback_reason)
            band_audit["psf_model"] = str(psf_fallback_model)
            if psf_fallback_model == "ncircular_gaussian":
                sigma = max(float(fwhm_pix) / 2.3548200450309493, 1e-3)
                ncomp = int(getattr(cfg, "ncircular_gaussian_n", 3))
                if ncomp == 1:
                    sigmas = [sigma]
                    weights = [1.0]
                elif ncomp == 2:
                    sigmas = [sigma, 2.5 * sigma]
                    weights = [0.85, 0.15]
                elif ncomp == 3:
                    sigmas = [sigma, 2.0 * sigma, 4.0 * sigma]
                    weights = [0.8, 0.15, 0.05]
                else:
                    raise ValueError(f"ncircular_gaussian_n must be 1, 2, or 3. Got: {ncomp}")
                psf_by_band[band] = NCircularGaussianPSF(sigmas, weights)
            elif psf_fallback_model == "moffat":
                psf_by_band[band] = MoffatPSF(
                    fwhm_pix=float(fwhm_pix),
                    beta=float(cfg.moffat_beta),
                    radius_pix=float(cfg.moffat_radius_pix),
                )
            else:
                stamp = _gaussian_stamp_from_fwhm(fwhm_pix=float(fwhm_pix), radius_pix=float(cfg.fallback_stamp_radius_pix))
                psf_by_band[band] = GaussianMixturePSF.fromStamp(stamp, N=max(1, int(cfg.gaussian_mixture_n)))
            log.info("[PSF] %s: %s at %s -> fallback %s", band, fallback_reason, str(epsf_path), str(psf_by_band[band]))
        psf_audit[band] = band_audit
    return psf_by_band, psf_audit


def build_tractor_images_from_per_filter_patch(
    *,
    per_filter_patch: dict,
    psf_by_filt: dict[str, Any],
    use_sky_sigma: bool = True,
) -> tuple[list[Image], list[dict[str, Any]]]:
    tractor_images: list[Image] = []
    image_data: list[dict[str, Any]] = []
    for filt, d in per_filter_patch.items():
        filt = str(filt)
        img = np.ascontiguousarray(d["img_scaled"], dtype=np.float32)
        sig = d["sigma_sky_scaled"] if use_sky_sigma else d["sigma_total_scaled"]
        sig = np.ascontiguousarray(sig, dtype=np.float32)
        bad = np.asarray(d.get("bad", ~np.isfinite(img)), dtype=bool)
        bad |= ~np.isfinite(img) | ~np.isfinite(sig) | (sig <= 0)

        invvar = np.zeros_like(img, dtype=np.float32)
        good = ~bad
        invvar[good] = 1.0 / np.maximum(sig[good] * sig[good], 1e-12)

        tim = Image(
            data=img,
            invvar=invvar,
            psf=psf_by_filt.get(filt, None),
            wcs=NullWCS(),
            sky=ConstantSky(0.0),
            photocal=FluxesPhotoCal(str(filt)),
            name=str(filt),
        )
        tractor_images.append(tim)
        image_data.append(
            dict(
                filter=str(filt),
                fp=d.get("fp", None),
                scale=float(d.get("scale", np.nan)) if d.get("scale", None) is not None else np.nan,
                data=img,
                invvar=invvar,
                sigma_used=sig,
                bad=bad,
            )
        )
    return tractor_images, image_data


def build_catalog_from_cat_patch(cat_patch: pd.DataFrame, tractor_images: list[Image], cfg: PatchRunConfig) -> list[Any]:
    if "x_pix_patch" not in cat_patch.columns or "y_pix_patch" not in cat_patch.columns:
        raise ValueError("cat_patch must contain columns: x_pix_patch, y_pix_patch")
    xs = np.asarray(cat_patch["x_pix_patch"], dtype=np.float32)
    ys = np.asarray(cat_patch["y_pix_patch"], dtype=np.float32)

    bandnames = [str(getattr(tim, "name", f"band{i}")) for i, tim in enumerate(tractor_images)]
    col_map = {str(c).strip().lower(): str(c) for c in cat_patch.columns}
    init_fluxes = [dict((bn, float(cfg.eps_flux)) for bn in bandnames) for _ in range(len(cat_patch))]

    for tim, bn in zip(tractor_images, bandnames):
        img = np.ascontiguousarray(tim.getImage(), dtype=np.float32)
        inv = np.ascontiguousarray(tim.getInvvar(), dtype=np.float32)
        good = np.isfinite(img) & np.isfinite(inv) & (inv > 0)
        err = np.full_like(img, np.inf, dtype=np.float32)
        err[good] = (1.0 / np.sqrt(inv[good])).astype(np.float32)
        mask = ~good
        f_ap, _ferr, _flag = sep.sum_circle(img, xs, ys, r=float(cfg.r_ap), err=err, mask=mask)
        f_ap = np.asarray(f_ap, dtype=np.float32)
        f_ap = np.where(np.isfinite(f_ap), f_ap, 0.0)

        flux_col = col_map.get(f"flux_{bn}".lower())
        if flux_col is not None:
            f_cat = pd.to_numeric(cat_patch[flux_col], errors="coerce").to_numpy(dtype=np.float32)
            use_cat = np.isfinite(f_cat) & (f_cat > 0)
            f = np.where(use_cat, f_cat, f_ap).astype(np.float32)
        else:
            f = f_ap
        f = np.maximum(f, float(cfg.eps_flux))
        for i in range(len(cat_patch)):
            init_fluxes[i][bn] = float(f[i])

    if "TYPE" in cat_patch.columns:
        t = cat_patch["TYPE"].astype(str).str.upper().to_numpy()
    else:
        t = np.full(len(cat_patch), str(cfg.gal_model).upper(), dtype=object)

    if "ELL" in cat_patch.columns:
        ell = pd.to_numeric(cat_patch["ELL"], errors="coerce").to_numpy(dtype=float)
        ab = np.clip(1.0 - np.nan_to_num(ell, nan=0.0), 0.1, 1.0)
    else:
        ab = np.full(len(cat_patch), 0.8, dtype=float)

    if "THETA" in cat_patch.columns:
        phi = pd.to_numeric(cat_patch["THETA"], errors="coerce").to_numpy(dtype=float)
        phi = np.nan_to_num(phi, nan=0.0)
        phi = -np.where(phi >= 0, 90 - phi, -90 - phi)
    else:
        phi = np.zeros(len(cat_patch), dtype=float)

    if "Re" in cat_patch.columns:
        re_ = pd.to_numeric(cat_patch["Re"], errors="coerce").to_numpy(dtype=float)
        re_ = np.where(np.isfinite(re_), re_, float(cfg.re_fallback_pix))
        re_ = np.clip(re_, 0.3, 100.0)
    else:
        re_ = np.full(len(cat_patch), float(cfg.re_fallback_pix), dtype=float)

    catalog: list[Any] = []
    for i in range(len(cat_patch)):
        pos = PixPos(float(xs[i]), float(ys[i]))
        bright = Fluxes(**init_fluxes[i])
        ti = str(t[i]).upper()
        if ti == "STAR":
            src = PointSource(pos, bright)
        else:
            RE_MIN = 0.1
            RE_MAX = 1000.0
            EE_MAX = 100.0
            shape = EllipseESoft.fromRAbPhi(float(re_[i]), float(ab[i]), float(phi[i]))
            shape.lowers = [float(np.log(RE_MIN)), -float(EE_MAX), -float(EE_MAX)]
            shape.uppers = [float(np.log(RE_MAX)), float(EE_MAX), float(EE_MAX)]

            if ti == "SERSIC":
                src = SersicGalaxy(pos, bright, shape, SersicIndex(float(cfg.sersic_n_init)))
            elif ti == "DEV":
                src = DevGalaxy(pos, bright, shape)
            elif ti == "EXP":
                src = ExpGalaxy(pos, bright, shape)
            else:
                gm = str(cfg.gal_model).lower()
                if gm == "star":
                    src = PointSource(pos, bright)
                elif gm == "sersic":
                    src = SersicGalaxy(pos, bright, shape, SersicIndex(float(cfg.sersic_n_init)))
                elif gm == "dev":
                    src = DevGalaxy(pos, bright, shape)
                else:
                    src = ExpGalaxy(pos, bright, shape)
        catalog.append(src)
    return catalog


def _robust_limits_percentile(arr: np.ndarray, p_lo: float, p_hi: float, min_span: float = 1e-6):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    vmin, vmax = np.percentile(a, [p_lo, p_hi])
    vmin, vmax = float(vmin), float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return None, None
    if (vmax - vmin) < min_span:
        vmin = float(np.min(a))
        vmax = float(np.max(a))
        if (vmax - vmin) < min_span:
            vmin -= 0.5
            vmax += 0.5
    return vmin, vmax


def _robust_resid_limits_percentile(resid: np.ndarray, p_abs: float = 99, min_span: float = 1e-6):
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None, None
    a = np.nanpercentile(np.abs(r), p_abs)
    if not np.isfinite(a) or a < min_span:
        a = np.nanmax(np.abs(r))
        if not np.isfinite(a) or a < min_span:
            a = 1.0
    return float(-a), float(+a)


def _true_minmax(arr: np.ndarray):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    return float(a.min()), float(a.max())


def _white_from_images(
    *,
    images: list[Image],
    model: bool,
    tr: Optional[Tractor] = None,
    model_images: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    num = None
    den = None
    for i, tim in enumerate(images):
        img = np.asarray(tim.getImage(), dtype=np.float32)
        inv = np.asarray(tim.getInvvar(), dtype=np.float32)
        good = np.isfinite(img) & np.isfinite(inv) & (inv > 0)
        w = np.where(good, inv, 0.0).astype(np.float32, copy=False)
        if model:
            if model_images is not None:
                m = np.asarray(model_images[i], dtype=np.float32)
            elif tr is not None:
                m = np.asarray(tr.getModelImage(i), dtype=np.float32)
            else:
                raise ValueError("tr or model_images is required when model=True")
            sky = float(getattr(getattr(tim, "sky", None), "getValue", lambda: 0.0)())
            img_use = m + sky
        else:
            img_use = img
        if num is None:
            num = w * img_use
            den = w.copy()
        else:
            num += w * img_use
            den += w
    if num is None or den is None:
        raise ValueError("No images provided for white")
    out = np.full_like(den, np.nan, dtype=np.float32)
    ok = den > 0
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def save_patch_overview(
    *,
    tr: Tractor,
    cat_patch: Optional[pd.DataFrame],
    meta: Optional[dict],
    outpath: Path,
    data_p_lo: float,
    data_p_hi: float,
    model_p_lo: float,
    model_p_hi: float,
    resid_p_abs: float,
    log: Optional[logging.Logger] = None,
    model_by_band: Optional[list[np.ndarray]] = None,
    per_band_fit_positions: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    meta = meta or {}
    patch_tag = str(meta.get("patch_tag", "patch"))
    nsrc = int(meta.get("n_sources", len(tr.catalog))) if hasattr(tr, "catalog") else 0

    white_data = _white_from_images(images=list(tr.images), model=False, tr=None)
    white_model = _white_from_images(images=list(tr.images), model=True, tr=tr, model_images=model_by_band)
    white_resid = np.asarray(white_data - white_model, dtype=np.float32)

    dvmin, dvmax = _robust_limits_percentile(white_data, p_lo=data_p_lo, p_hi=data_p_hi)
    mvmin, mvmax = _robust_limits_percentile(white_model, p_lo=model_p_lo, p_hi=model_p_hi)
    rvmin, rvmax = _robust_resid_limits_percentile(white_resid, p_abs=resid_p_abs)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("Data (white)", white_data, dvmin, dvmax, "Data"),
        ("Model+Sky (white)", white_model, mvmin, mvmax, "Model+Sky"),
        ("Residual (white)", white_resid, rvmin, rvmax, "Residual"),
    ]

    x_orig = y_orig = None
    if isinstance(cat_patch, pd.DataFrame) and "x_pix_patch" in cat_patch.columns and "y_pix_patch" in cat_patch.columns:
        x_orig = pd.to_numeric(cat_patch["x_pix_patch"], errors="coerce").to_numpy(dtype=float)
        y_orig = pd.to_numeric(cat_patch["y_pix_patch"], errors="coerce").to_numpy(dtype=float)

    for ax, (title, arr, vmin, vmax, cblab) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if x_orig is not None and y_orig is not None:
            ok = np.isfinite(x_orig) & np.isfinite(y_orig)
            if np.any(ok):
                ax.scatter(x_orig[ok], y_orig[ok], s=18, marker="x", c="orange", linewidths=0.8, alpha=0.8)

        if per_band_fit_positions is not None:
            for bx_all, by_all in per_band_fit_positions:
                okf = np.isfinite(bx_all) & np.isfinite(by_all)
                if np.any(okf):
                    ax.scatter(bx_all[okf], by_all[okf], s=12, marker="x",
                               c="deepskyblue", linewidths=0.5, alpha=0.5)
                if x_orig is not None and y_orig is not None:
                    both_ok = ok & okf
                    for j in np.where(both_ok)[0]:
                        ax.plot([x_orig[j], bx_all[j]], [y_orig[j], by_all[j]],
                                color="deepskyblue", lw=0.3, alpha=0.4)
        else:
            x_fit = np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).x)
                              for s in tr.catalog], dtype=float)
            y_fit = np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).y)
                              for s in tr.catalog], dtype=float)
            okf = np.isfinite(x_fit) & np.isfinite(y_fit)
            if np.any(okf):
                ax.scatter(x_fit[okf], y_fit[okf], s=18, marker="x",
                           c="deepskyblue", linewidths=0.8, alpha=0.8)
            if x_orig is not None and y_orig is not None:
                both_ok = ok & okf
                for j in np.where(both_ok)[0]:
                    ax.plot([x_orig[j], x_fit[j]], [y_orig[j], y_fit[j]],
                            color="deepskyblue", lw=0.4, alpha=0.6)

        amin, amax = _true_minmax(arr)
        if amin is not None and amax is not None:
            sm = ScalarMappable(norm=Normalize(vmin=amin, vmax=amax), cmap=im.cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, extend="neither")
            cb.set_label(cblab)

    fig.suptitle(f"patch={patch_tag} | n_sources={nsrc}\norig=x orange, fit=x deepskyblue", fontsize=12)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=_PATCH_PLOT_DPI)
    plt.close(fig)
    if log:
        log.info("Wrote patch overview: %s", str(outpath))


def _cut(arr: np.ndarray, cx: float, cy: float, size: int):
    ix = int(round(float(cx)))
    iy = int(round(float(cy)))
    half = max(1, int(size) // 2)
    ny, nx = arr.shape
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))

    hx = min(half, ix, (nx - 1 - ix))
    hy = min(half, iy, (ny - 1 - iy))

    x0 = ix - hx
    x1 = ix + hx + 1
    y0 = iy - hy
    y1 = iy + hy + 1
    return arr[y0:y1, x0:x1], (y0, y1, x0, x1)


def save_source_montages(
    *,
    tr: Tractor,
    cat_patch: Optional[pd.DataFrame] = None,
    meta: Optional[dict] = None,
    outdir: Path,
    size: int,
    per_band_scale: bool,
    max_sources: Optional[int],
    start: int,
    data_p_lo: float,
    data_p_hi: float,
    model_p_lo: float,
    model_p_hi: float,
    resid_p_abs: float,
    log: Optional[logging.Logger] = None,
    model_by_band: Optional[list[np.ndarray]] = None,
    per_band_fit_positions: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    _ensure_dir(outdir)
    meta = meta or {}
    patch_tag = str(meta.get("patch_tag", "patch"))
    roi_bbox = meta.get("roi_bbox", None)
    x0_roi = y0_roi = None
    if isinstance(roi_bbox, (list, tuple)) and len(roi_bbox) == 4:
        try:
            x0_roi = float(roi_bbox[0])
            y0_roi = float(roi_bbox[2])
        except Exception:
            x0_roi = y0_roi = None

    ids = None
    types = None
    x_orig_all = y_orig_all = None
    xw_orig_all = yw_orig_all = None
    if isinstance(cat_patch, pd.DataFrame):
        if "ID" in cat_patch.columns:
            ids = cat_patch["ID"].astype(str).to_numpy()
        if "TYPE" in cat_patch.columns:
            types = cat_patch["TYPE"].astype(str).to_numpy()
        if "x_pix_patch" in cat_patch.columns and "y_pix_patch" in cat_patch.columns:
            x_orig_all = pd.to_numeric(cat_patch["x_pix_patch"], errors="coerce").to_numpy(dtype=float)
            y_orig_all = pd.to_numeric(cat_patch["y_pix_patch"], errors="coerce").to_numpy(dtype=float)
        if "x_pix_white" in cat_patch.columns and "y_pix_white" in cat_patch.columns:
            xw_orig_all = pd.to_numeric(cat_patch["x_pix_white"], errors="coerce").to_numpy(dtype=float)
            yw_orig_all = pd.to_numeric(cat_patch["y_pix_white"], errors="coerce").to_numpy(dtype=float)

    bandnames = [str(getattr(tim, "name", f"band{i}")) for i, tim in enumerate(tr.images)]

    data_by_band = [np.asarray(tim.getImage(), dtype=np.float32) for tim in tr.images]
    if model_by_band is None:
        model_by_band = [np.asarray(tr.getModelImage(i), dtype=np.float32) for i in range(len(tr.images))]
    sky_by_band = [float(getattr(getattr(tim, "sky", None), "getValue", lambda: 0.0)()) for tim in tr.images]

    indices = list(range(len(tr.catalog)))
    indices = [i for i in indices if i >= int(start)]
    if max_sources is not None:
        indices = indices[: int(max_sources)]

    if log:
        log.info("Saving %d montage(s) to %s", len(indices), str(outdir))

    for si in indices:
        src = tr.catalog[si]
        pos = src.getPosition() if hasattr(src, "getPosition") else src.pos

        if per_band_fit_positions is not None and x_orig_all is not None and y_orig_all is not None:
            cx = float(x_orig_all[si]) if (si < len(x_orig_all) and np.isfinite(x_orig_all[si])) else float(pos.x)
            cy = float(y_orig_all[si]) if (si < len(y_orig_all) and np.isfinite(y_orig_all[si])) else float(pos.y)
        else:
            cx, cy = float(pos.x), float(pos.y)

        sid = None
        if ids is not None and si < len(ids):
            sid = str(ids[si])
        stype = None
        if types is not None and si < len(types):
            stype = str(types[si])

        x0_fit_white = None
        y0_fit_white = None
        if x0_roi is not None and y0_roi is not None and np.isfinite(cx) and np.isfinite(cy):
            x0_fit_white = float(cx + x0_roi)
            y0_fit_white = float(cy + y0_roi)

        x0_orig = y0_orig = None
        if x_orig_all is not None and y_orig_all is not None and si < len(x_orig_all):
            x0_orig = float(x_orig_all[si]) if np.isfinite(x_orig_all[si]) else None
            y0_orig = float(y_orig_all[si]) if np.isfinite(y_orig_all[si]) else None

        x0_orig_white = y0_orig_white = None
        if xw_orig_all is not None and yw_orig_all is not None and si < len(xw_orig_all):
            x0_orig_white = float(xw_orig_all[si]) if np.isfinite(xw_orig_all[si]) else None
            y0_orig_white = float(yw_orig_all[si]) if np.isfinite(yw_orig_all[si]) else None

        x_fit_all = np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).x) for s in tr.catalog], dtype=float)
        y_fit_all = np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).y) for s in tr.catalog], dtype=float)

        ncols = len(bandnames)
        nrows = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), constrained_layout=True)
        if ncols == 1:
            axes = np.array(axes).reshape(nrows, 1)

        axes[0, 0].set_ylabel("Data", fontsize=11)
        axes[1, 0].set_ylabel("Model+Sky", fontsize=11)
        axes[2, 0].set_ylabel("Residual", fontsize=11)

        for i, bn in enumerate(bandnames):
            data, (y0, y1, x0, x1) = _cut(data_by_band[i], cx, cy, size)
            model, _ = _cut(model_by_band[i], cx, cy, size)
            model = model + sky_by_band[i]
            resid = data - model

            if per_band_fit_positions is not None:
                bx_all, by_all = per_band_fit_positions[i]
                x_fit_c = float(bx_all[si]) - float(x0)
                y_fit_c = float(by_all[si]) - float(y0)
                x_fit_c_all = bx_all - float(x0)
                y_fit_c_all = by_all - float(y0)
            else:
                x_fit_c = cx - float(x0)
                y_fit_c = cy - float(y0)
                x_fit_c_all = x_fit_all - float(x0)
                y_fit_c_all = y_fit_all - float(y0)
            x_orig_c = (x0_orig - float(x0)) if (x0_orig is not None) else None
            y_orig_c = (y0_orig - float(y0)) if (y0_orig is not None) else None
            ok_fit = (
                np.isfinite(x_fit_c_all)
                & np.isfinite(y_fit_c_all)
                & (x_fit_c_all >= 0)
                & (x_fit_c_all < (x1 - x0))
                & (y_fit_c_all >= 0)
                & (y_fit_c_all < (y1 - y0))
            )
            ok_fit_others = ok_fit.copy()
            if 0 <= si < len(ok_fit_others):
                ok_fit_others[si] = False

            ok_orig = None
            ok_orig_others = None
            x_orig_c_all = y_orig_c_all = None
            if x_orig_all is not None and y_orig_all is not None:
                x_orig_c_all = x_orig_all - float(x0)
                y_orig_c_all = y_orig_all - float(y0)
                ok_orig = (
                    np.isfinite(x_orig_c_all)
                    & np.isfinite(y_orig_c_all)
                    & (x_orig_c_all >= 0)
                    & (x_orig_c_all < (x1 - x0))
                    & (y_orig_c_all >= 0)
                    & (y_orig_c_all < (y1 - y0))
                )
                ok_orig_others = ok_orig.copy()
                if 0 <= si < len(ok_orig_others):
                    ok_orig_others[si] = False

            if per_band_scale:
                dvmin, dvmax = _robust_limits_percentile(data, p_lo=data_p_lo, p_hi=data_p_hi)
                mvmin, mvmax = _robust_limits_percentile(model, p_lo=model_p_lo, p_hi=model_p_hi)
                rvmin, rvmax = _robust_resid_limits_percentile(resid, p_abs=resid_p_abs)
            else:
                dvmin = dvmax = mvmin = mvmax = rvmin = rvmax = None

            _has_fit = np.isfinite(x_fit_c) and np.isfinite(y_fit_c)
            _has_orig = (x_orig_c is not None) and (y_orig_c is not None) and np.isfinite(x_orig_c) and np.isfinite(y_orig_c)
            _others_connect = (
                ok_fit_others is not None and ok_orig_others is not None
                and x_orig_c_all is not None and y_orig_c_all is not None
            )

            ax = axes[0, i]
            im0 = ax.imshow(data, origin="lower", cmap="gray", vmin=dvmin, vmax=dvmax, interpolation="nearest")
            ax.set_title(bn, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if _has_fit:
                ax.axvline(x_fit_c, color="magenta", lw=0.8, alpha=0.9)
                ax.axhline(y_fit_c, color="magenta", lw=0.8, alpha=0.9)
            if _has_orig:
                ax.axvline(x_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
                ax.axhline(y_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
            if np.any(ok_fit_others):
                ax.scatter(
                    x_fit_c_all[ok_fit_others],
                    y_fit_c_all[ok_fit_others],
                    s=18,
                    marker="x",
                    c="deepskyblue",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if ok_orig_others is not None and np.any(ok_orig_others):
                ax.scatter(
                    x_orig_c_all[ok_orig_others],
                    y_orig_c_all[ok_orig_others],
                    s=18,
                    marker="x",
                    c="orange",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if _others_connect:
                for j in np.where(ok_fit_others & ok_orig_others)[0]:
                    ax.plot([x_orig_c_all[j], x_fit_c_all[j]], [y_orig_c_all[j], y_fit_c_all[j]],
                            color="cyan", lw=0.4, alpha=0.5)

            dmin_true, dmax_true = _true_minmax(data)
            if dmin_true is not None and dmax_true is not None:
                sm0 = ScalarMappable(norm=Normalize(vmin=dmin_true, vmax=dmax_true), cmap=im0.cmap)
                sm0.set_array([])
                cb0 = fig.colorbar(sm0, ax=ax, fraction=0.046, pad=0.02, extend="neither")
                cb0.set_label("Data")

            ax = axes[1, i]
            im1 = ax.imshow(model, origin="lower", cmap="gray", vmin=mvmin, vmax=mvmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if _has_fit:
                ax.axvline(x_fit_c, color="magenta", lw=0.8, alpha=0.9)
                ax.axhline(y_fit_c, color="magenta", lw=0.8, alpha=0.9)
            if _has_orig:
                ax.axvline(x_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
                ax.axhline(y_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
            if np.any(ok_fit_others):
                ax.scatter(
                    x_fit_c_all[ok_fit_others],
                    y_fit_c_all[ok_fit_others],
                    s=18,
                    marker="x",
                    c="deepskyblue",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if ok_orig_others is not None and np.any(ok_orig_others):
                ax.scatter(
                    x_orig_c_all[ok_orig_others],
                    y_orig_c_all[ok_orig_others],
                    s=18,
                    marker="x",
                    c="orange",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if _others_connect:
                for j in np.where(ok_fit_others & ok_orig_others)[0]:
                    ax.plot([x_orig_c_all[j], x_fit_c_all[j]], [y_orig_c_all[j], y_fit_c_all[j]],
                            color="cyan", lw=0.4, alpha=0.5)

            mmin_true, mmax_true = _true_minmax(model)
            if mmin_true is not None and mmax_true is not None:
                sm1 = ScalarMappable(norm=Normalize(vmin=mmin_true, vmax=mmax_true), cmap=im1.cmap)
                sm1.set_array([])
                cb1 = fig.colorbar(sm1, ax=ax, fraction=0.046, pad=0.02, extend="neither")
                cb1.set_label("Model+Sky")

            ax = axes[2, i]
            im2 = ax.imshow(resid, origin="lower", cmap="gray", vmin=rvmin, vmax=rvmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if _has_fit:
                ax.axvline(x_fit_c, color="magenta", lw=0.8, alpha=0.9)
                ax.axhline(y_fit_c, color="magenta", lw=0.8, alpha=0.9)
            if _has_orig:
                ax.axvline(x_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
                ax.axhline(y_orig_c, color="lime", lw=0.8, alpha=0.9, linestyle="--")
            if np.any(ok_fit_others):
                ax.scatter(
                    x_fit_c_all[ok_fit_others],
                    y_fit_c_all[ok_fit_others],
                    s=18,
                    marker="x",
                    c="deepskyblue",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if ok_orig_others is not None and np.any(ok_orig_others):
                ax.scatter(
                    x_orig_c_all[ok_orig_others],
                    y_orig_c_all[ok_orig_others],
                    s=18,
                    marker="x",
                    c="orange",
                    linewidths=0.8,
                    alpha=0.8,
                )
            if _others_connect:
                for j in np.where(ok_fit_others & ok_orig_others)[0]:
                    ax.plot([x_orig_c_all[j], x_fit_c_all[j]], [y_orig_c_all[j], y_fit_c_all[j]],
                            color="cyan", lw=0.4, alpha=0.5)

            rmin_true, rmax_true = _true_minmax(resid)
            if rmin_true is not None and rmax_true is not None:
                sm2 = ScalarMappable(norm=Normalize(vmin=rmin_true, vmax=rmax_true), cmap=im2.cmap)
                sm2.set_array([])
                cb2 = fig.colorbar(sm2, ax=ax, fraction=0.046, pad=0.02, extend="neither")
                cb2.set_label("Residual")

        s_id = sid if sid is not None else f"row{si}"
        s0 = f"ID={s_id} | patch={patch_tag} | idx_in_patch={si}/{len(tr.catalog)-1}"
        if stype is not None and stype != "" and stype.lower() != "nan":
            s0 += f" | TYPE={stype}"
        s_legend = "fit=center crosshair=magenta, orig=crosshair=lime(--), fit/others=x deepskyblue, orig/others=x orange"
        title = s0 + "\n" + s_legend
        fig.suptitle(title, fontsize=12)
        out_path = outdir / f"src_{si:06d}.png"
        plt.savefig(out_path, dpi=_PATCH_PLOT_DPI)
        plt.close(fig)


def compute_flux_errors(tr: Tractor) -> dict[tuple[int, str], float]:
    names = tr.getParamNames()
    bandnames = [str(getattr(tim, "name", f"band{i}")) for i, tim in enumerate(tr.images)]
    band_to_i = {bn: i for i, bn in enumerate(bandnames)}

    pat = re.compile(r"^catalog\.source(\d+)\.brightness\.(.+)$")

    var = tr.optimize(variance=True, just_variance=True)
    if var is None or len(var) != len(names):
        return {}

    out: dict[tuple[int, str], float] = {}
    for i, pname in enumerate(names):
        m = pat.match(pname)
        if not m:
            continue
        sid = int(m.group(1))
        bn = m.group(2)
        if band_to_i.get(bn, None) is None:
            continue
        v = var[i]
        if v is None or (not np.isfinite(v)) or v <= 0:
            continue
        out[(sid, bn)] = float(math.sqrt(v))
    return out


def _get_sersic_n(src, default=np.nan):
    if not hasattr(src, "sersicindex"):
        return default
    si = src.sersicindex
    if hasattr(si, "getValue"):
        return float(si.getValue())
    if hasattr(si, "value"):
        return float(si.value)
    if hasattr(si, "n"):
        return float(si.n)
    return default


def _get_shape_params(src):
    if not hasattr(src, "shape"):
        return (np.nan, np.nan, np.nan)
    sh = src.shape
    re_ = float(getattr(sh, "re", np.nan))
    ab_ = float(getattr(sh, "ab", np.nan))
    if hasattr(sh, "phi"):
        phi_deg = float(getattr(sh, "phi", np.nan))
    elif hasattr(sh, "theta"):
        try:
            phi_deg = float(-np.degrees(sh.theta))
        except Exception:
            phi_deg = np.nan
    else:
        phi_deg = np.nan
    if np.isfinite(phi_deg):
        phi_deg = float(phi_deg % 180.0)
    return (re_, ab_, phi_deg)


def _extract_source_params_to_df(
    out: pd.DataFrame,
    i: int,
    src: Any,
    col_suffix: str,
    x0_roi: float,
    y0_roi: float,
) -> None:
    """Extract fitted position/morphology for one source into the output DataFrame.

    col_suffix: "" for multi-band (shared columns), "_{band}" for single-band.
    """
    pos = src.getPosition() if hasattr(src, "getPosition") else src.pos
    out.at[i, f"x_pix_patch{col_suffix}_fit"] = float(pos.x)
    out.at[i, f"y_pix_patch{col_suffix}_fit"] = float(pos.y)
    out.at[i, f"x_pix_white{col_suffix}_fit"] = float(pos.x) + float(x0_roi)
    out.at[i, f"y_pix_white{col_suffix}_fit"] = float(pos.y) + float(y0_roi)

    if isinstance(src, SersicGalaxy):
        out.at[i, "stype_fit"] = "sersic"
        out.at[i, f"sersic_n{col_suffix}_fit"] = _get_sersic_n(src)
    elif isinstance(src, DevGalaxy):
        out.at[i, "stype_fit"] = "dev"
    elif isinstance(src, ExpGalaxy):
        out.at[i, "stype_fit"] = "exp"
    else:
        out.at[i, "stype_fit"] = "star"
        return

    re_, ab_, phi_ = _get_shape_params(src)
    out.at[i, f"re_pix{col_suffix}_fit"] = re_
    out.at[i, f"ab{col_suffix}_fit"] = ab_
    out.at[i, f"phi_deg{col_suffix}_fit"] = phi_
    if np.isfinite(re_):
        out.at[i, f"Re{col_suffix}_fit"] = float(re_)
    if np.isfinite(ab_):
        out.at[i, f"ELL{col_suffix}_fit"] = float(1.0 - ab_)
    if np.isfinite(phi_):
        out.at[i, f"THETA{col_suffix}_fit"] = float(phi_ - 90.0)


def _freeze_gaussian_mixture_psfs(tr: Tractor, log: logging.Logger) -> list[str]:
    """Freeze GaussianMixturePSF image params to avoid ConstrainedOptimizer bounds mismatch."""
    frozen: list[str] = []
    for tim in tr.images:
        try:
            if isinstance(getattr(tim, "psf", None), GaussianMixturePSF):
                tim.freezeParam("psf")
                frozen.append(str(getattr(tim, "name", "")))
        except Exception as e:
            log.warning("[PSF] Failed to freeze GaussianMixturePSF params for %s: %s",
                        str(getattr(tim, "name", "")), str(e))
    if frozen:
        log.warning("[PSF] Frozen GaussianMixturePSF image params for optimizer stability in bands: %s",
                    ",".join(frozen))
    return frozen


def _run_optimizer(
    tr: Tractor,
    cfg: PatchRunConfig,
    log: logging.Logger,
    log_prefix: str = "",
) -> tuple[int, bool, bool, float | None]:
    """Run the Tractor optimization loop. Returns (niters, converged, hit_max_iters, last_dlnp)."""
    converged = False
    last_dlnp: float | None = None
    niters = 0
    pfx = f"[{log_prefix}] " if log_prefix else ""
    for it in range(int(cfg.n_opt_iters)):
        dlnp, _X, _alpha = tr.optimize()
        niters = it + 1
        try:
            last_dlnp = float(dlnp)
        except Exception:
            last_dlnp = None
        log.info("%sopt iter %02d  dlnp=%s", pfx, it, str(dlnp))
        if dlnp < float(cfg.dlnp_stop):
            converged = True
            break
    margin = max(0, int(getattr(cfg, "flag_maxiter_margin", 0)))
    hit_max_iters = (not converged) and (niters >= max(1, int(cfg.n_opt_iters) - margin))
    if hit_max_iters:
        log.info("%sWARNING: did not converge. niters=%d n_opt_iters=%d margin=%d last_dlnp=%s",
                 pfx, int(niters), int(cfg.n_opt_iters), int(margin), str(last_dlnp))
    return niters, converged, hit_max_iters, last_dlnp


def _compute_psf_audit_summary(psf_audit: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Derive PSF audit summary fields from per-band psf_audit."""
    low_star_bands = sorted(
        b for b, a in psf_audit.items()
        if isinstance(a.get("fallback_reason", None), str)
        and str(a["fallback_reason"]).startswith("low-star ePSF")
    )
    fallback_bands = sorted(b for b, a in psf_audit.items() if not bool(a.get("used_epsf", False)))
    fallback_reason_by_band = {
        b: a.get("fallback_reason", None)
        for b, a in psf_audit.items() if not bool(a.get("used_epsf", False))
    }
    used_epsf_bands = sorted(b for b, a in psf_audit.items() if bool(a.get("used_epsf", False)))
    return dict(
        low_star_bands=low_star_bands,
        fallback_bands=fallback_bands,
        fallback_reason_by_band=fallback_reason_by_band,
        used_epsf_bands=used_epsf_bands,
    )


def _populate_psf_audit_columns(
    out: pd.DataFrame,
    cfg: PatchRunConfig,
    audit: dict[str, Any],
    psf_frozen_bands: list[str],
    fallback_reason_by_band: dict[str, Any],
) -> None:
    """Write PSF audit columns into the output DataFrame."""
    out["psf_min_epsf_nstars_for_use"] = int(getattr(cfg, "min_epsf_nstars_for_use", 5))
    out["psf_used_epsf_band_count"] = int(len(audit["used_epsf_bands"]))
    out["psf_fallback_band_count"] = int(len(audit["fallback_bands"]))
    out["psf_low_star_band_count"] = int(len(audit["low_star_bands"]))
    out["psf_fallback_bands"] = ",".join(audit["fallback_bands"])
    out["psf_low_star_bands"] = ",".join(audit["low_star_bands"])
    out["psf_frozen_bands_for_optimizer"] = ",".join(sorted(b for b in psf_frozen_bands if b))
    out["psf_fallback_reasons_json"] = json.dumps(fallback_reason_by_band, sort_keys=True)


def _build_visualization_tractor(
    *,
    per_filter_patch: dict,
    psf_by_band: dict[str, Any],
    all_band_tractors: dict[str, Tractor],
    bandnames: list[str],
    n_sources: int,
    cfg: PatchRunConfig,
    log: logging.Logger,
) -> Tractor | None:
    """Build a combined multi-band Tractor for visualization after single-band fitting.

    Uses the first band's fitted position/shape as reference and each band's fitted
    flux.  The model will be exact for the reference band; for other bands positions
    may differ by < 1 pix which is negligible for diagnostic plots.
    """
    try:
        all_images, _ = build_tractor_images_from_per_filter_patch(
            per_filter_patch=per_filter_patch, psf_by_filt=psf_by_band, use_sky_sigma=True)

        ref_band = bandnames[0]
        vis_catalog: list[Any] = []
        for i in range(n_sources):
            ref_src = all_band_tractors[ref_band].catalog[i]
            ref_pos = ref_src.getPosition() if hasattr(ref_src, "getPosition") else ref_src.pos
            pos = PixPos(float(ref_pos.x), float(ref_pos.y))

            fluxes_dict: dict[str, float] = {}
            for bn in bandnames:
                try:
                    bx = all_band_tractors[bn].catalog[i].getBrightness()
                    fluxes_dict[bn] = float(bx.getFlux(bn))
                except Exception:
                    fluxes_dict[bn] = float(cfg.eps_flux)
            bright = Fluxes(**fluxes_dict)

            if isinstance(ref_src, SersicGalaxy):
                re_, ab_, phi_ = _get_shape_params(ref_src)
                shape = EllipseESoft.fromRAbPhi(
                    max(float(re_), 0.1) if np.isfinite(re_) else float(cfg.re_fallback_pix),
                    float(ab_) if np.isfinite(ab_) else 0.8,
                    float(phi_) if np.isfinite(phi_) else 0.0,
                )
                sn_val = _get_sersic_n(ref_src, default=float(cfg.sersic_n_init))
                vis_catalog.append(SersicGalaxy(pos, bright, shape, SersicIndex(sn_val)))
            elif isinstance(ref_src, DevGalaxy):
                re_, ab_, phi_ = _get_shape_params(ref_src)
                shape = EllipseESoft.fromRAbPhi(
                    max(float(re_), 0.1) if np.isfinite(re_) else float(cfg.re_fallback_pix),
                    float(ab_) if np.isfinite(ab_) else 0.8,
                    float(phi_) if np.isfinite(phi_) else 0.0,
                )
                vis_catalog.append(DevGalaxy(pos, bright, shape))
            elif isinstance(ref_src, ExpGalaxy):
                re_, ab_, phi_ = _get_shape_params(ref_src)
                shape = EllipseESoft.fromRAbPhi(
                    max(float(re_), 0.1) if np.isfinite(re_) else float(cfg.re_fallback_pix),
                    float(ab_) if np.isfinite(ab_) else 0.8,
                    float(phi_) if np.isfinite(phi_) else 0.0,
                )
                vis_catalog.append(ExpGalaxy(pos, bright, shape))
            else:
                vis_catalog.append(PointSource(pos, bright))

        return Tractor(all_images, vis_catalog)
    except Exception as e:
        log.warning("Failed to build visualization Tractor: %s", str(e))
        return None


def run_patch(*, payload: dict, epsf_root: Path, outdir: Path, cfg: PatchRunConfig) -> pd.DataFrame:
    global _PATCH_PLOT_DPI
    _PATCH_PLOT_DPI = int(getattr(cfg, "plot_dpi", 150))

    meta = payload.get("meta", {}) or {}
    tag = _patch_tag_from_meta(meta)
    patch_out = outdir / tag
    _ensure_dir(patch_out)
    _ensure_dir(patch_out / "cutouts")

    log = _make_logger(patch_out, tag)
    log.info("Starting patch run: %s  (multi_band_simultaneous=%s)", tag,
             str(cfg.enable_multi_band_simultaneous_fitting))

    (x0_roi, x1_roi, y0_roi, y1_roi) = _roi_bbox_from_meta(meta)
    meta = dict(meta)
    meta.setdefault("roi_bbox", (int(x0_roi), int(x1_roi), int(y0_roi), int(y1_roi)))
    (patch_out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    per_filter_patch = payload["per_filter_patch"]
    cat_patch = payload["cat_patch"]
    if not isinstance(cat_patch, pd.DataFrame):
        cat_patch = pd.DataFrame(cat_patch)

    epsf_tag = str(meta.get("epsf_tag", ""))
    if not epsf_tag:
        raise ValueError("meta.epsf_tag is required")

    psf_by_band, psf_audit = build_psf_by_band_for_patch(
        per_filter_patch=per_filter_patch,
        epsf_root=epsf_root,
        epsf_tag=epsf_tag,
        cfg=cfg,
        log=log,
    )

    audit = _compute_psf_audit_summary(psf_audit)
    meta = dict(meta)
    meta["psf_min_epsf_nstars_for_use"] = int(getattr(cfg, "min_epsf_nstars_for_use", 5))
    meta["psf_used_epsf_band_count"] = int(len(audit["used_epsf_bands"]))
    meta["psf_fallback_band_count"] = int(len(audit["fallback_bands"]))
    meta["psf_low_star_band_count"] = int(len(audit["low_star_bands"]))
    meta["psf_fallback_bands"] = audit["fallback_bands"]
    meta["psf_low_star_bands"] = audit["low_star_bands"]
    meta["psf_audit"] = psf_audit
    meta["enable_multi_band_simultaneous_fitting"] = bool(cfg.enable_multi_band_simultaneous_fitting)

    # ------------------------------------------------------------------
    #  Multi-band simultaneous fitting (default / original behaviour)
    # ------------------------------------------------------------------
    if cfg.enable_multi_band_simultaneous_fitting:
        tractor_images, _image_data = build_tractor_images_from_per_filter_patch(
            per_filter_patch=per_filter_patch, psf_by_filt=psf_by_band, use_sky_sigma=True)
        catalog = build_catalog_from_cat_patch(cat_patch, tractor_images, cfg)

        tr = Tractor(tractor_images, catalog, optimizer=ConstrainedOptimizer())
        tr.thawAllRecursive()
        psf_frozen_bands = _freeze_gaussian_mixture_psfs(tr, log)

        niters, converged, hit_max_iters, last_dlnp = _run_optimizer(tr, cfg, log)

        meta["psf_frozen_bands_for_optimizer"] = sorted(b for b in psf_frozen_bands if b)
        meta["opt_niters"] = int(niters)
        meta["opt_converged"] = bool(converged)
        meta["opt_hit_max_iters"] = bool(hit_max_iters)
        meta["opt_last_dlnp"] = float(last_dlnp) if last_dlnp is not None and np.isfinite(last_dlnp) else None
        (patch_out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        out = cat_patch.copy()
        out["patch_tag"] = str(tag)
        out["epsf_tag"] = str(epsf_tag)
        _populate_psf_audit_columns(out, cfg, audit, psf_frozen_bands, audit["fallback_reason_by_band"])
        out["opt_niters"] = int(niters)
        out["opt_converged"] = bool(converged)
        out["opt_hit_max_iters"] = bool(hit_max_iters)
        out["opt_last_dlnp"] = float(last_dlnp) if last_dlnp is not None and np.isfinite(last_dlnp) else np.nan
        out["x_pix_patch_fit"] = np.nan
        out["y_pix_patch_fit"] = np.nan
        out["x_pix_white_fit"] = np.nan
        out["y_pix_white_fit"] = np.nan
        out["RA_fit"] = np.nan
        out["DEC_fit"] = np.nan
        out["stype_fit"] = ""
        out["sersic_n_fit"] = np.nan
        out["re_pix_fit"] = np.nan
        out["ab_fit"] = np.nan
        out["phi_deg_fit"] = np.nan
        out["ELL_fit"] = np.nan
        out["Re_fit"] = np.nan
        out["THETA_fit"] = np.nan

        bandnames = [str(getattr(tim, "name", f"band{i}")) for i, tim in enumerate(tractor_images)]
        for bn in bandnames:
            out[f"FLUX_{bn}_fit"] = np.nan
            out[f"FLUXERR_{bn}_fit"] = np.nan

        for i, src in enumerate(tr.catalog):
            bx = src.getBrightness() if hasattr(src, "getBrightness") else src.brightness
            _extract_source_params_to_df(out, i, src, "", x0_roi, y0_roi)
            for bn in bandnames:
                try:
                    out.at[i, f"FLUX_{bn}_fit"] = float(bx.getFlux(bn))
                except Exception:
                    out.at[i, f"FLUX_{bn}_fit"] = np.nan

        try:
            ferr = compute_flux_errors(tr)
            for (sid, bn), fe in ferr.items():
                if 0 <= sid < len(out):
                    out.at[sid, f"FLUXERR_{bn}_fit"] = float(fe)
            log.info("Computed flux errors for %d (source,band) pairs", len(ferr))
        except Exception as e:
            log.info("WARNING: flux error computation failed: %s", str(e))

        out_csv = patch_out / f"{tag}_cat_fit.csv"
        out.to_csv(out_csv, index=False)
        log.info("Wrote fitted catalog: %s", str(out_csv))

        if cfg.save_cutouts:
            save_source_montages(
                tr=tr, cat_patch=cat_patch, meta=meta,
                outdir=patch_out / "cutouts",
                size=int(cfg.cutout_size_pix), per_band_scale=True,
                max_sources=cfg.cutout_max_sources, start=int(cfg.cutout_start),
                data_p_lo=float(cfg.cutout_data_p_lo), data_p_hi=float(cfg.cutout_data_p_hi),
                model_p_lo=float(cfg.cutout_model_p_lo), model_p_hi=float(cfg.cutout_model_p_hi),
                resid_p_abs=float(cfg.cutout_resid_p_abs), log=log,
            )

        if getattr(cfg, "save_patch_overview", False):
            try:
                save_patch_overview(
                    tr=tr, cat_patch=cat_patch, meta=meta,
                    outpath=patch_out / "patch_overview.png",
                    data_p_lo=float(cfg.cutout_data_p_lo), data_p_hi=float(cfg.cutout_data_p_hi),
                    model_p_lo=float(cfg.cutout_model_p_lo), model_p_hi=float(cfg.cutout_model_p_hi),
                    resid_p_abs=float(cfg.cutout_resid_p_abs), log=log,
                )
            except Exception as e:
                log.info("WARNING: patch overview failed: %s", str(e))

        log.info("Done patch run: %s", tag)
        return out

    # ------------------------------------------------------------------
    #  Single-band independent fitting
    # ------------------------------------------------------------------
    bandnames = sorted(str(b) for b in per_filter_patch.keys())
    log.info("Single-band fitting mode: fitting %d bands independently: %s",
             len(bandnames), ", ".join(bandnames))

    out = cat_patch.copy()
    out["patch_tag"] = str(tag)
    out["epsf_tag"] = str(epsf_tag)
    _populate_psf_audit_columns(out, cfg, audit, [], audit["fallback_reason_by_band"])
    out["stype_fit"] = ""

    for bn in bandnames:
        out[f"x_pix_patch_{bn}_fit"] = np.nan
        out[f"y_pix_patch_{bn}_fit"] = np.nan
        out[f"x_pix_white_{bn}_fit"] = np.nan
        out[f"y_pix_white_{bn}_fit"] = np.nan
        out[f"RA_{bn}_fit"] = np.nan
        out[f"DEC_{bn}_fit"] = np.nan
        out[f"sersic_n_{bn}_fit"] = np.nan
        out[f"re_pix_{bn}_fit"] = np.nan
        out[f"ab_{bn}_fit"] = np.nan
        out[f"phi_deg_{bn}_fit"] = np.nan
        out[f"Re_{bn}_fit"] = np.nan
        out[f"ELL_{bn}_fit"] = np.nan
        out[f"THETA_{bn}_fit"] = np.nan
        out[f"FLUX_{bn}_fit"] = np.nan
        out[f"FLUXERR_{bn}_fit"] = np.nan
        out[f"opt_niters_{bn}"] = 0
        out[f"opt_converged_{bn}"] = False
        out[f"opt_hit_max_iters_{bn}"] = False
        out[f"opt_last_dlnp_{bn}"] = np.nan

    all_psf_frozen: list[str] = []
    all_band_tractors: dict[str, Tractor] = {}

    for bn in bandnames:
        log.info("=== Fitting band: %s ===", bn)

        single_band_patch = {bn: per_filter_patch[bn]}
        band_images, _ = build_tractor_images_from_per_filter_patch(
            per_filter_patch=single_band_patch, psf_by_filt=psf_by_band, use_sky_sigma=True)
        band_catalog = build_catalog_from_cat_patch(cat_patch, band_images, cfg)

        band_tr = Tractor(band_images, band_catalog, optimizer=ConstrainedOptimizer())
        band_tr.thawAllRecursive()
        frozen = _freeze_gaussian_mixture_psfs(band_tr, log)
        all_psf_frozen.extend(frozen)

        niters, converged, hit_max_iters, last_dlnp = _run_optimizer(band_tr, cfg, log, log_prefix=bn)

        out[f"opt_niters_{bn}"] = int(niters)
        out[f"opt_converged_{bn}"] = bool(converged)
        out[f"opt_hit_max_iters_{bn}"] = bool(hit_max_iters)
        out[f"opt_last_dlnp_{bn}"] = float(last_dlnp) if last_dlnp is not None and np.isfinite(last_dlnp) else np.nan

        for i, src in enumerate(band_tr.catalog):
            bx = src.getBrightness() if hasattr(src, "getBrightness") else src.brightness
            _extract_source_params_to_df(out, i, src, f"_{bn}", x0_roi, y0_roi)
            try:
                out.at[i, f"FLUX_{bn}_fit"] = float(bx.getFlux(bn))
            except Exception:
                out.at[i, f"FLUX_{bn}_fit"] = np.nan

        try:
            ferr = compute_flux_errors(band_tr)
            for (sid, b), fe in ferr.items():
                if 0 <= sid < len(out):
                    out.at[sid, f"FLUXERR_{b}_fit"] = float(fe)
            log.info("[%s] Computed flux errors for %d source(s)", bn, len(ferr))
        except Exception as e:
            log.info("[%s] WARNING: flux error computation failed: %s", bn, str(e))

        all_band_tractors[bn] = band_tr

    out["psf_frozen_bands_for_optimizer"] = ",".join(sorted(set(b for b in all_psf_frozen if b)))
    meta["psf_frozen_bands_for_optimizer"] = sorted(set(b for b in all_psf_frozen if b))
    (patch_out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    out_csv = patch_out / f"{tag}_cat_fit.csv"
    out.to_csv(out_csv, index=False)
    log.info("Wrote fitted catalog: %s", str(out_csv))

    if cfg.save_cutouts or getattr(cfg, "save_patch_overview", False):
        vis_tr = _build_visualization_tractor(
            per_filter_patch=per_filter_patch,
            psf_by_band=psf_by_band,
            all_band_tractors=all_band_tractors,
            bandnames=bandnames,
            n_sources=len(cat_patch),
            cfg=cfg,
            log=log,
        )
        if vis_tr is not None:
            exact_models = [
                np.asarray(all_band_tractors[bn].getModelImage(0), dtype=np.float32)
                for bn in bandnames
            ]
            per_band_positions = [
                (
                    np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).x)
                              for s in all_band_tractors[bn].catalog], dtype=float),
                    np.array([float((s.getPosition() if hasattr(s, "getPosition") else s.pos).y)
                              for s in all_band_tractors[bn].catalog], dtype=float),
                )
                for bn in bandnames
            ]
            if cfg.save_cutouts:
                save_source_montages(
                    tr=vis_tr, cat_patch=cat_patch, meta=meta,
                    outdir=patch_out / "cutouts",
                    size=int(cfg.cutout_size_pix), per_band_scale=True,
                    max_sources=cfg.cutout_max_sources, start=int(cfg.cutout_start),
                    data_p_lo=float(cfg.cutout_data_p_lo), data_p_hi=float(cfg.cutout_data_p_hi),
                    model_p_lo=float(cfg.cutout_model_p_lo), model_p_hi=float(cfg.cutout_model_p_hi),
                    resid_p_abs=float(cfg.cutout_resid_p_abs), log=log,
                    model_by_band=exact_models,
                    per_band_fit_positions=per_band_positions,
                )
            if getattr(cfg, "save_patch_overview", False):
                try:
                    save_patch_overview(
                        tr=vis_tr, cat_patch=cat_patch, meta=meta,
                        outpath=patch_out / "patch_overview.png",
                        data_p_lo=float(cfg.cutout_data_p_lo), data_p_hi=float(cfg.cutout_data_p_hi),
                        model_p_lo=float(cfg.cutout_model_p_lo), model_p_hi=float(cfg.cutout_model_p_hi),
                        resid_p_abs=float(cfg.cutout_resid_p_abs), log=log,
                        model_by_band=exact_models,
                        per_band_fit_positions=per_band_positions,
                    )
                except Exception as e:
                    log.info("WARNING: patch overview failed: %s", str(e))

    log.info("Done patch run: %s", tag)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Path to patch input .pkl.gz")
    ap.add_argument("--epsf-root", required=True, help="ePSF root directory")
    ap.add_argument("--outdir", required=True, help="Output base directory")
    ap.add_argument("--no-enable-multi-band-simultaneous-fitting", action="store_true",
                    help="Fit each band independently instead of simultaneous multi-band fitting")
    ap.add_argument("--no-cutouts", action="store_true", help="Disable montage saving")
    ap.add_argument("--no-patch-overview", action="store_true", help="Disable per-patch overview plot")
    ap.add_argument("--cutout-size", type=int, default=100)
    ap.add_argument("--cutout-max-sources", type=int, default=None)
    ap.add_argument("--cutout-start", type=int, default=0)
    ap.add_argument("--cutout-data-p-lo", type=float, default=1)
    ap.add_argument("--cutout-data-p-hi", type=float, default=99)
    ap.add_argument("--cutout-model-p-lo", type=float, default=1)
    ap.add_argument("--cutout-model-p-hi", type=float, default=99)
    ap.add_argument("--cutout-resid-p-abs", type=float, default=99)
    ap.add_argument("--gal-model", default="dev", choices=["dev", "exp", "sersic", "star"])
    ap.add_argument("--n-opt-iters", type=int, default=200)
    ap.add_argument("--dlnp-stop", type=float, default=1e-6)
    ap.add_argument("--flag-maxiter-margin", type=int, default=5)
    ap.add_argument("--r-ap", type=float, default=5.0)
    ap.add_argument("--eps-flux", type=float, default=1e-4)
    ap.add_argument("--psf-model", default="hybrid", choices=["hybrid", "pixelized", "gaussian_mixture_from_stamp"])
    ap.add_argument("--psf-fallback-model", default="gaussian_mixture", choices=["gaussian_mixture", "ncircular_gaussian", "moffat"])
    ap.add_argument("--min-epsf-nstars-for-use", type=int, default=5)
    ap.add_argument("--ncircular-gaussian-n", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--hybrid-psf-n", type=int, default=4)
    ap.add_argument("--gaussian-mixture-n", type=int, default=4)
    ap.add_argument("--sersic-n-init", type=float, default=3.0)
    ap.add_argument("--re-fallback-pix", type=float, default=3.0)
    ap.add_argument("--fallback-default-fwhm-pix", type=float, default=4.0)
    ap.add_argument("--fallback-stamp-radius-pix", type=float, default=25.0)
    ap.add_argument("--moffat-beta", type=float, default=3.5)
    ap.add_argument("--moffat-radius-pix", type=float, default=25.0)
    ap.add_argument("--plot-dpi", type=int, default=150)
    args = ap.parse_args(argv)

    payload = load_patch_inputs(args.inputs)
    cfg = PatchRunConfig(
        enable_multi_band_simultaneous_fitting=not args.no_enable_multi_band_simultaneous_fitting,
        save_cutouts=not args.no_cutouts,
        save_patch_overview=not args.no_patch_overview,
        cutout_size_pix=int(args.cutout_size),
        cutout_max_sources=args.cutout_max_sources,
        cutout_start=int(args.cutout_start),
        cutout_data_p_lo=float(args.cutout_data_p_lo),
        cutout_data_p_hi=float(args.cutout_data_p_hi),
        cutout_model_p_lo=float(args.cutout_model_p_lo),
        cutout_model_p_hi=float(args.cutout_model_p_hi),
        cutout_resid_p_abs=float(args.cutout_resid_p_abs),
        gal_model=str(args.gal_model),
        n_opt_iters=int(args.n_opt_iters),
        dlnp_stop=float(args.dlnp_stop),
        flag_maxiter_margin=int(args.flag_maxiter_margin),
        r_ap=float(args.r_ap),
        eps_flux=float(args.eps_flux),
        psf_model=str(args.psf_model),
        psf_fallback_model=str(args.psf_fallback_model),
        min_epsf_nstars_for_use=int(args.min_epsf_nstars_for_use),
        ncircular_gaussian_n=int(args.ncircular_gaussian_n),
        hybrid_psf_n=int(args.hybrid_psf_n),
        gaussian_mixture_n=int(args.gaussian_mixture_n),
        sersic_n_init=float(args.sersic_n_init),
        re_fallback_pix=float(args.re_fallback_pix),
        fallback_default_fwhm_pix=float(args.fallback_default_fwhm_pix),
        fallback_stamp_radius_pix=float(args.fallback_stamp_radius_pix),
        moffat_beta=float(args.moffat_beta),
        moffat_radius_pix=float(args.moffat_radius_pix),
        plot_dpi=int(args.plot_dpi),
    )

    run_patch(payload=payload, epsf_root=Path(args.epsf_root), outdir=Path(args.outdir), cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
