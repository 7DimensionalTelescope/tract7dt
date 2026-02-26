from __future__ import annotations

import json
import logging
import os
import multiprocessing as mp
import queue as queue_mod
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import concurrent.futures as cf
import numpy as np

import sep
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.psf import EPSFBuilder, extract_stars
from photutils.psf.epsf_stars import EPSFStar, EPSFStars

logger = logging.getLogger("tract7dt.epsf")


sep.set_sub_object_limit(200000)
sep.set_extract_pixstack(1000000)

_EPSF_IMAGE_DICT = None
_EPSF_CFG = None
_EPSF_PLOT_DPI = 150
_EPSF_GAIAXP = None
_EPSF_USE_GAIAXP = True
_EPSF_OUT_ROOT = None
_EPSF_ACTIVE_PATCHES: set[str] | None = None


@dataclass
class EPSFConfig:
    epsf_ngrid: int = 7
    ngrid: int = 6
    thresh_sigma: float = 10.0
    minarea: int = 25
    cutout_size: int = 61
    edge_pad: int = 10
    min_sep: float = 30.0
    max_stars: int = 30
    q_range: tuple[float, float] = (0.7, 1.3)
    psfstar_mode: str = "sep+gaia"
    gaia_mag_min: float = 10.0
    gaia_mag_max: float = 30.0
    gaia_snap_to_sep: bool = True
    gaia_forced_k_fwhm: float = 1.5
    gaia_forced_rmin_pix: float = 3.0
    gaia_forced_rmax_pix: float = 15.0
    gaia_snr_min: float = 20.0
    gaia_match_r_pix: float = 5.0
    gaia_seed_sat_r: int = 5
    gaia_seed_sat_div: float = 1.3
    oversamp: int = 1
    maxiters: int = 30
    do_clip: bool = True
    recenter_boxsize: int = 9
    final_psf_size: int = 55
    save_star_montage_max: int = 100
    background_subtract_patch: bool = True
    background_boxsize: int = 96
    background_filtersize: int = 9
    background_source_mask_sigma: float = 3.0
    background_mask_dilate: int = 1
    star_local_bkg_subtract: bool = True
    star_local_bkg_annulus_rin_pix: float = 20.0
    star_local_bkg_annulus_rout_pix: float = 30.0
    star_local_bkg_sigma: float = 3.0
    star_local_bkg_minpix: int = 30
    save_patch_background_diagnostics: bool = True
    save_star_local_background_diagnostics: bool = True
    diagnostics_show_colorbar: bool = True
    save_growth_curve: bool = True
    save_residual_diagnostics: bool = True
    residual_diag_max_stars: int = 100


def _grid_bounds(nx: int, ny: int, ngrid: int, r: int, c: int) -> tuple[int, int, int, int]:
    if not (0 <= r < ngrid and 0 <= c < ngrid):
        raise ValueError("r/c out of range")
    x_step = nx // ngrid
    y_step = ny // ngrid
    x0 = c * x_step
    x1 = nx if c == ngrid - 1 else (c + 1) * x_step
    y0 = r * y_step
    y1 = ny if r == ngrid - 1 else (r + 1) * y_step
    return x0, x1, y0, y1


def _dilate_mask(mask: np.ndarray, n_iter: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(n_iter))):
        m = out.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        out[1:, 1:] |= m[:-1, :-1]
        out[1:, :-1] |= m[:-1, 1:]
        out[:-1, 1:] |= m[1:, :-1]
        out[:-1, :-1] |= m[1:, 1:]
    return out


def _estimate_patch_background(
    *,
    img: np.ndarray,
    sigma: np.ndarray,
    bad: np.ndarray,
    cfg: EPSFConfig,
):
    arr = np.ascontiguousarray(img, dtype=np.float32)
    sig = np.ascontiguousarray(sigma, dtype=np.float32)
    m_bad = np.ascontiguousarray(bad.astype(bool), dtype=np.bool_)
    m_work = m_bad.copy()

    good = (~m_work) & np.isfinite(arr) & np.isfinite(sig) & (sig > 0)
    if np.any(good):
        med = float(np.nanmedian(arr[good]))
        sig_med = float(np.nanmedian(sig[good]))
        if np.isfinite(sig_med) and sig_med > 0:
            src = good & (arr > med + float(cfg.background_source_mask_sigma) * sig_med)
            if np.any(src):
                m_work |= _dilate_mask(src, n_iter=int(cfg.background_mask_dilate))

    try:
        bkg = sep.Background(
            arr,
            mask=np.ascontiguousarray(m_work, dtype=np.bool_),
            bw=max(8, int(cfg.background_boxsize)),
            bh=max(8, int(cfg.background_boxsize)),
            fw=max(1, int(cfg.background_filtersize)),
            fh=max(1, int(cfg.background_filtersize)),
        )
        bkg_map = np.asarray(bkg.back(), dtype=np.float32)
        bkg_rms = np.asarray(bkg.rms(), dtype=np.float32)
        out = arr - bkg_map
        stats = dict(
            patch_bkg_median=float(np.nanmedian(bkg_map)),
            patch_bkg_mean=float(np.nanmean(bkg_map)),
            patch_bkg_rms_median=float(np.nanmedian(bkg_rms)),
            patch_bkg_rms_mean=float(np.nanmean(bkg_rms)),
            patch_bkg_mask_frac=float(np.mean(m_work)),
            patch_bkg_status="ok",
        )
        return out, bkg_map, bkg_rms, stats
    except Exception as e:
        z = np.zeros_like(arr, dtype=np.float32)
        stats = dict(
            patch_bkg_median=0.0,
            patch_bkg_mean=0.0,
            patch_bkg_rms_median=np.nan,
            patch_bkg_rms_mean=np.nan,
            patch_bkg_mask_frac=float(np.mean(m_work)),
            patch_bkg_status="error",
            patch_bkg_error=f"{type(e).__name__}: {e}",
        )
        return arr.copy(), z, z, stats


def _subtract_local_background_from_stars(*, stars, cfg: EPSFConfig):
    if stars is None:
        return None, []
    star_list = list(stars.all_stars) if hasattr(stars, "all_stars") else list(stars)
    if len(star_list) == 0:
        return stars, []

    out_stars: list[EPSFStar] = []
    local_stats: list[dict] = []
    sigclip = SigmaClip(sigma=float(cfg.star_local_bkg_sigma), maxiters=5)

    for st in star_list:
        data = np.asarray(st.data, dtype=np.float32)
        weights = np.asarray(getattr(st, "weights", np.ones_like(data)), dtype=np.float32)
        ny, nx = data.shape

        center = getattr(st, "cutout_center", None)
        if center is None:
            cx = float((nx - 1) / 2.0)
            cy = float((ny - 1) / 2.0)
            cutout_center = None
        else:
            cx = float(center[0])
            cy = float(center[1])
            cutout_center = (cx, cy)

        max_r = max(2.0, min((nx - 1) / 2.0, (ny - 1) / 2.0) - 1.0)
        rout = float(np.clip(float(cfg.star_local_bkg_annulus_rout_pix), 2.0, max_r))
        rin = float(np.clip(float(cfg.star_local_bkg_annulus_rin_pix), 0.0, max(0.0, rout - 1.0)))

        yy, xx = np.indices(data.shape, dtype=np.float32)
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ann = (rr2 >= rin * rin) & (rr2 <= rout * rout)
        valid = ann & np.isfinite(data) & np.isfinite(weights) & (weights > 0)
        vals = data[valid]

        local_bkg = 0.0
        status = "fallback_zero"
        if vals.size >= int(cfg.star_local_bkg_minpix):
            clipped = sigclip(vals, masked=True)
            if hasattr(clipped, "compressed"):
                vv = np.asarray(clipped.compressed(), dtype=np.float32)
            else:
                vv = np.asarray(clipped, dtype=np.float32)
            vv = vv[np.isfinite(vv)]
            if vv.size >= int(cfg.star_local_bkg_minpix):
                local_bkg = float(np.nanmedian(vv))
                status = "ok"

        data_sub = data - float(local_bkg)
        origin_raw = np.asarray(getattr(st, "origin", (0, 0))).astype(int).tolist()
        if len(origin_raw) >= 2:
            origin = (int(origin_raw[0]), int(origin_raw[1]))
        else:
            origin = (0, 0)

        st_new = EPSFStar(
            data_sub,
            weights=weights,
            cutout_center=cutout_center,
            origin=origin,
            wcs_large=getattr(st, "wcs_large", None),
            id_label=getattr(st, "id_label", None),
        )
        out_stars.append(st_new)
        local_stats.append(
            dict(
                id_label=str(getattr(st, "id_label", "")),
                local_bkg=float(local_bkg),
                annulus_npix=int(vals.size),
                annulus_rin=float(rin),
                annulus_rout=float(rout),
                status=status,
            )
        )

    return EPSFStars(out_stars), local_stats


def _select_psf_stars_sep(
    *,
    img: np.ndarray,
    sigma: np.ndarray,
    bad: np.ndarray,
    cfg: EPSFConfig,
    saturate: float | None = None,
    seed_tbl: Table | None = None,
    mode: str = "sep+gaia",
    forced_r_pix: float | None = None,
) -> Table:
    img = np.ascontiguousarray(img, dtype=np.float32)
    sig = np.ascontiguousarray(sigma, dtype=np.float32)
    bad = np.ascontiguousarray(bad.astype(bool), dtype=np.bool_)

    ny, nx = img.shape
    need = int(cfg.cutout_size // 2 + cfg.edge_pad)

    try:
        objs = sep.extract(
            img,
            thresh=float(cfg.thresh_sigma),
            err=sig,
            mask=bad,
            minarea=int(cfg.minarea),
            deblend_nthresh=32,
            deblend_cont=0.001,
            clean=True,
            clean_param=1.0,
        )
    except Exception:
        objs = None

    cand: list[tuple[float, float, float]] = []
    cand_xy: list[tuple[float, float]] = []
    cand_q: list[float] = []

    has_ab = (objs is not None) and (len(objs) > 0) and ("a" in objs.dtype.names) and ("b" in objs.dtype.names)

    if objs is not None and len(objs) > 0:
        has_peak = ("peak" in objs.dtype.names)
        has_flux = ("flux" in objs.dtype.names)

        for o in objs:
            x = float(o["x"])
            y = float(o["y"])
            if not (need < x < nx - need - 1 and need < y < ny - need - 1):
                continue

            peak = float(o["peak"]) if has_peak else np.nan
            if saturate is not None and np.isfinite(saturate) and np.isfinite(peak):
                if peak >= float(saturate) / float(cfg.gaia_seed_sat_div):
                    continue

            q = np.nan
            if has_ab:
                a = float(o["a"])
                b = float(o["b"])
                if b <= 0:
                    continue
                q = a / b
                if not (cfg.q_range[0] <= q <= cfg.q_range[1]):
                    continue

            score = float(o["flux"]) if has_flux else (peak if np.isfinite(peak) else 0.0)
            cand.append((x, y, score))
            cand_xy.append((x, y))
            cand_q.append(float(q) if np.isfinite(q) else np.nan)

    cand.sort(key=lambda t: t[2], reverse=True)

    match_r2 = float(cfg.gaia_match_r_pix) ** 2

    def _has_sep_near(x0: float, y0: float) -> bool:
        for cx, cy in cand_xy:
            dx = cx - x0
            dy = cy - y0
            if dx * dx + dy * dy <= match_r2:
                return True
        return False

    def _seed_local_max_ok(x0: float, y0: float) -> bool:
        if saturate is None or (not np.isfinite(saturate)):
            return True
        r = int(cfg.gaia_seed_sat_r)
        ix = int(np.round(x0))
        iy = int(np.round(y0))
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return False
        xlo = max(0, ix - r)
        xhi = min(nx, ix + r + 1)
        ylo = max(0, iy - r)
        yhi = min(ny, iy + r + 1)

        sub = img[ylo:yhi, xlo:xhi]
        sub_bad = bad[ylo:yhi, xlo:xhi]
        if sub.size == 0:
            return False
        if np.all(sub_bad):
            return False
        local_max = np.nanmax(np.where(sub_bad, -np.inf, sub))
        return bool(local_max < float(saturate) / float(cfg.gaia_seed_sat_div))

    picked_xy: list[tuple[float, float]] = []
    picked_origin: list[str] = []

    mode = str(mode).strip().lower()
    if mode not in ("sep", "sep+gaia", "gaia"):
        mode = "sep+gaia"

    snap = bool(cfg.gaia_snap_to_sep) and (mode != "sep")
    snapped_sep_used: set[int] = set()

    def _nearest_sep_within(x0: float, y0: float) -> tuple[int, float, float] | None:
        if not snap:
            return None
        best_j = None
        best_d2 = match_r2
        best_xy = None
        for j, (cx, cy) in enumerate(cand_xy):
            if j in snapped_sep_used:
                continue
            dx = cx - x0
            dy = cy - y0
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_j = j
                best_xy = (cx, cy)
        if best_j is not None and best_xy is not None:
            snapped_sep_used.add(best_j)
            return (best_j, float(best_xy[0]), float(best_xy[1]))
        return None

    if mode in ("sep+gaia", "gaia") and (seed_tbl is not None) and len(seed_tbl) > 0:
        for sx, sy in zip(seed_tbl["x"], seed_tbl["y"]):
            sx = float(sx)
            sy = float(sy)
            if not (need < sx < nx - need - 1 and need < sy < ny - need - 1):
                continue
            if bad[int(np.round(sy)), int(np.round(sx))]:
                continue
            if not _seed_local_max_ok(sx, sy):
                continue

            snapped = _nearest_sep_within(sx, sy)
            snapped_j = None
            if snapped is not None:
                snapped_j, px, py = snapped
                origin = "both"

                if snapped_j is not None and (snapped_j < len(cand_q)):
                    q = cand_q[snapped_j]
                    if np.isfinite(q) and (not (cfg.q_range[0] <= q <= cfg.q_range[1])):
                        continue
            else:
                px, py = sx, sy
                origin = "both" if _has_sep_near(sx, sy) else "gaia"

            try:
                r = float(forced_r_pix) if (forced_r_pix is not None) else np.nan
                if not np.isfinite(r) or r <= 0:
                    r = 3.0

                flux, fluxerr, flag = sep.sum_circle(img, [px], [py], r, err=sig, mask=bad)
                snr = float(flux[0]) / float(fluxerr[0]) if (fluxerr[0] > 0) else -np.inf
                if (int(flag[0]) != 0) or (not np.isfinite(snr)) or (snr < float(cfg.gaia_snr_min)):
                    continue
            except Exception:
                continue

            if all((px - ox) ** 2 + (py - oy) ** 2 >= (cfg.min_sep**2) for ox, oy in picked_xy):
                picked_xy.append((px, py))
                picked_origin.append(origin)
            if len(picked_xy) >= int(cfg.max_stars):
                break

    if mode in ("sep", "sep+gaia"):
        for x, y, _ in cand:
            if all((x - px) ** 2 + (y - py) ** 2 >= (cfg.min_sep**2) for px, py in picked_xy):
                picked_xy.append((x, y))
                picked_origin.append("sep")
            if len(picked_xy) >= int(cfg.max_stars):
                break

    if len(picked_xy) == 0:
        return Table(names=("x", "y", "id", "origin"), dtype=("f8", "f8", "i8", "U8"))

    arr = np.array(picked_xy, dtype=np.float32)
    out = Table()
    out["x"] = arr[:, 0]
    out["y"] = arr[:, 1]
    out["id"] = np.arange(len(out), dtype=np.int64)
    out["origin"] = np.array(picked_origin, dtype="U8")
    return out


def _sep_extract_candidates_full(
    *,
    img_full: np.ndarray,
    sigma_full: np.ndarray,
    bad_full: np.ndarray,
    cfg: EPSFConfig,
    saturate: float | None = None,
):
    img_full = np.ascontiguousarray(img_full, dtype=np.float32)
    sig_full = np.ascontiguousarray(sigma_full, dtype=np.float32)
    bad_full = np.ascontiguousarray(bad_full.astype(bool), dtype=np.bool_)

    try:
        objs = sep.extract(
            img_full,
            thresh=float(cfg.thresh_sigma),
            err=sig_full,
            mask=bad_full,
            minarea=int(cfg.minarea),
            deblend_nthresh=32,
            deblend_cont=0.001,
            clean=True,
            clean_param=1.0,
        )
    except Exception:
        return None

    if objs is None or len(objs) == 0:
        return None

    names = set(getattr(objs.dtype, "names", []) or [])
    has_peak = "peak" in names
    has_flux = "flux" in names
    has_ab = ("a" in names) and ("b" in names)

    x = np.asarray(objs["x"], dtype=float)
    y = np.asarray(objs["y"], dtype=float)

    peak = np.asarray(objs["peak"], dtype=float) if has_peak else np.full(len(objs), np.nan)
    flux = np.asarray(objs["flux"], dtype=float) if has_flux else np.full(len(objs), np.nan)

    if has_ab:
        a = np.asarray(objs["a"], dtype=float)
        b = np.asarray(objs["b"], dtype=float)
        q = np.where(b > 0, a / b, np.nan)
    else:
        q = np.full(len(objs), np.nan)

    score = np.where(np.isfinite(flux), flux, np.where(np.isfinite(peak), peak, 0.0))

    m = np.isfinite(x) & np.isfinite(y)
    if saturate is not None and np.isfinite(saturate):
        m &= (~np.isfinite(peak)) | (peak < float(saturate) / float(cfg.gaia_seed_sat_div))

    if np.any(np.isfinite(q)):
        qok = (~np.isfinite(q)) | ((q >= float(cfg.q_range[0])) & (q <= float(cfg.q_range[1])))
        m &= qok

    x = x[m]
    y = y[m]
    score = score[m]
    q = q[m]

    if x.size > 1:
        o = np.argsort(score)[::-1]
        x = x[o]
        y = y[o]
        score = score[o]
        q = q[o]

    return x, y, score, q


def _select_psf_stars_from_candidates(
    *,
    img: np.ndarray,
    sigma: np.ndarray,
    bad: np.ndarray,
    cfg: EPSFConfig,
    cand: list[tuple[float, float, float]],
    cand_xy: list[tuple[float, float]],
    cand_q: list[float],
    saturate: float | None = None,
    seed_tbl: Table | None = None,
    mode: str = "sep+gaia",
    forced_r_pix: float | None = None,
) -> Table:
    img = np.ascontiguousarray(img, dtype=np.float32)
    sig = np.ascontiguousarray(sigma, dtype=np.float32)
    bad = np.ascontiguousarray(bad.astype(bool), dtype=np.bool_)

    ny, nx = img.shape
    need = int(cfg.cutout_size // 2 + cfg.edge_pad)

    if cand is None or len(cand) == 0:
        cand = []
        cand_xy = []
        cand_q = []

    match_r2 = float(cfg.gaia_match_r_pix) ** 2

    def _has_sep_near(x0: float, y0: float) -> bool:
        for cx, cy in cand_xy:
            dx = cx - x0
            dy = cy - y0
            if dx * dx + dy * dy <= match_r2:
                return True
        return False

    def _seed_local_max_ok(x0: float, y0: float) -> bool:
        if saturate is None or (not np.isfinite(saturate)):
            return True
        r = int(cfg.gaia_seed_sat_r)
        ix = int(np.round(x0))
        iy = int(np.round(y0))
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return False
        xlo = max(0, ix - r)
        xhi = min(nx, ix + r + 1)
        ylo = max(0, iy - r)
        yhi = min(ny, iy + r + 1)

        sub = img[ylo:yhi, xlo:xhi]
        sub_bad = bad[ylo:yhi, xlo:xhi]
        if sub.size == 0:
            return False
        if np.all(sub_bad):
            return False
        local_max = np.nanmax(np.where(sub_bad, -np.inf, sub))
        return bool(local_max < float(saturate) / float(cfg.gaia_seed_sat_div))

    picked_xy: list[tuple[float, float]] = []
    picked_origin: list[str] = []

    mode = str(mode).strip().lower()
    if mode not in ("sep", "sep+gaia", "gaia"):
        mode = "sep+gaia"

    snap = bool(cfg.gaia_snap_to_sep) and (mode != "sep")
    snapped_sep_used: set[int] = set()

    def _nearest_sep_within(x0: float, y0: float) -> tuple[int, float, float] | None:
        if not snap:
            return None
        best_j = None
        best_d2 = match_r2
        best_xy = None
        for j, (cx, cy) in enumerate(cand_xy):
            if j in snapped_sep_used:
                continue
            dx = cx - x0
            dy = cy - y0
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_j = j
                best_xy = (cx, cy)
        if best_j is not None and best_xy is not None:
            snapped_sep_used.add(best_j)
            return (best_j, float(best_xy[0]), float(best_xy[1]))
        return None

    if mode in ("sep+gaia", "gaia") and (seed_tbl is not None) and len(seed_tbl) > 0:
        for sx, sy in zip(seed_tbl["x"], seed_tbl["y"]):
            sx = float(sx)
            sy = float(sy)
            if not (need < sx < nx - need - 1 and need < sy < ny - need - 1):
                continue
            if bad[int(np.round(sy)), int(np.round(sx))]:
                continue
            if not _seed_local_max_ok(sx, sy):
                continue

            snapped = _nearest_sep_within(sx, sy)
            snapped_j = None
            if snapped is not None:
                snapped_j, px, py = snapped
                origin = "both"

                if snapped_j is not None and (snapped_j < len(cand_q)):
                    q = cand_q[snapped_j]
                    if np.isfinite(q) and (not (cfg.q_range[0] <= q <= cfg.q_range[1])):
                        continue
            else:
                px, py = sx, sy
                origin = "both" if _has_sep_near(sx, sy) else "gaia"

            try:
                r = float(forced_r_pix) if (forced_r_pix is not None) else np.nan
                if not np.isfinite(r) or r <= 0:
                    r = 3.0

                flux, fluxerr, flag = sep.sum_circle(img, [px], [py], r, err=sig, mask=bad)
                snr = float(flux[0]) / float(fluxerr[0]) if (fluxerr[0] > 0) else -np.inf
                if (int(flag[0]) != 0) or (not np.isfinite(snr)) or (snr < float(cfg.gaia_snr_min)):
                    continue
            except Exception:
                continue

            if all((px - ox) ** 2 + (py - oy) ** 2 >= (cfg.min_sep**2) for ox, oy in picked_xy):
                picked_xy.append((px, py))
                picked_origin.append(origin)
            if len(picked_xy) >= int(cfg.max_stars):
                break

    if mode in ("sep", "sep+gaia"):
        for x, y, _ in cand:
            if all((x - px) ** 2 + (y - py) ** 2 >= (cfg.min_sep**2) for px, py in picked_xy):
                picked_xy.append((x, y))
                picked_origin.append("sep")
            if len(picked_xy) >= int(cfg.max_stars):
                break

    if len(picked_xy) == 0:
        return Table(names=("x", "y", "id", "origin"), dtype=("f8", "f8", "i8", "U8"))

    arr = np.array(picked_xy, dtype=np.float32)
    out = Table()
    out["x"] = arr[:, 0]
    out["y"] = arr[:, 1]
    out["id"] = np.arange(len(out), dtype=np.int64)
    out["origin"] = np.array(picked_origin, dtype="U8")
    return out


def _build_epsf_from_invvar(*, img: np.ndarray, invvar: np.ndarray, star_tbl: Table, cfg: EPSFConfig):
    if star_tbl is None or len(star_tbl) == 0:
        return None

    img = np.ascontiguousarray(img, dtype=np.float32)
    invvar = np.ascontiguousarray(invvar, dtype=np.float32)

    id_to_origin: dict[str, str] = {}
    if ("id" in star_tbl.colnames) and ("origin" in star_tbl.colnames):
        for i, o in zip(star_tbl["id"], star_tbl["origin"]):
            try:
                id_to_origin[str(i)] = str(o)
            except Exception:
                pass

    err = np.zeros_like(img, dtype=np.float32)
    m = np.isfinite(invvar) & (invvar > 0)
    if not np.any(m):
        return None
    err[m] = (1.0 / np.sqrt(invvar[m])).astype(np.float32)
    err[~m] = float(np.median(err[m]))

    nd = NDData(img, uncertainty=StdDevUncertainty(err))
    stars = extract_stars(nd, star_tbl, size=int(cfg.cutout_size))
    if stars is None or len(stars) == 0:
        return None
    stars_raw = stars
    local_bkg_stats = []
    if bool(getattr(cfg, "star_local_bkg_subtract", True)):
        stars_corr, local_bkg_stats = _subtract_local_background_from_stars(stars=stars, cfg=cfg)
        if stars_corr is not None and len(stars_corr) > 0:
            stars = stars_corr
    stars_bgsub = stars

    sigclip = SigmaClip(sigma=5.0) if cfg.do_clip else SigmaClip(sigma=np.inf)
    epsf_builder = EPSFBuilder(
        oversampling=int(cfg.oversamp),
        maxiters=int(cfg.maxiters),
        recentering_boxsize=int(cfg.recenter_boxsize),
        recentering_maxiters=30,
        sigma_clip=sigclip,
        progress_bar=False,
    )

    try:
        epsf, fitted = epsf_builder(stars)
    except Exception as e:
        return dict(
            epsf_arr=None,
            epsf_model=None,
            nstars_extract=int(len(stars)),
            nstars_used=0,
            stars_all=stars,
            stars_all_raw=stars_raw,
            stars_all_bgsub=stars_bgsub,
            stars_used=None,
            id_to_origin=id_to_origin,
            local_bkg_stats=local_bkg_stats,
            error=f"{type(e).__name__}: {e}",
        )

    arr = np.asarray(epsf.data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    H, W = arr.shape
    crop = int(cfg.final_psf_size)
    if crop < min(H, W):
        y0c = (H - crop) // 2
        x0c = (W - crop) // 2
        arr = arr[y0c : y0c + crop, x0c : x0c + crop]

    s = float(arr.sum())
    if s > 0:
        arr /= s

    n_extract = int(len(stars))
    n_used = int(len(fitted)) if fitted is not None else 0

    return dict(
        epsf_arr=arr,
        epsf_model=epsf,
        nstars_extract=n_extract,
        nstars_used=n_used,
        stars_all=stars,
        stars_all_raw=stars_raw,
        stars_all_bgsub=stars_bgsub,
        stars_used=fitted,
        id_to_origin=id_to_origin,
        local_bkg_stats=local_bkg_stats,
        epsf_qa=_epsf_growth_metrics(arr),
    )


def _save_star_overlay(img: np.ndarray, star_tbl: Table, outpath: Path, title: str, dpi: int = 150):
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)
    m = np.isfinite(img)
    if np.any(m):
        vmin, vmax = np.percentile(img[m], [5, 99])
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if star_tbl is None or len(star_tbl) == 0:
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(outpath, dpi=dpi)
        plt.close(fig)
        return

    origins = np.asarray(star_tbl["origin"]).astype(str) if "origin" in star_tbl.colnames else np.array(["unknown"] * len(star_tbl))

    m_sep = origins == "sep"
    if np.any(m_sep):
        ax.scatter(
            star_tbl["x"][m_sep],
            star_tbl["y"][m_sep],
            s=55,
            facecolors="none",
            edgecolors="orange",
            linewidths=1.2,
            marker="o",
            label=f"sep-only (N={int(np.sum(m_sep))})",
        )

    m_gaia = origins == "gaia"
    if np.any(m_gaia):
        ax.scatter(
            star_tbl["x"][m_gaia],
            star_tbl["y"][m_gaia],
            s=80,
            c="cyan",
            linewidths=1.6,
            marker="+",
            label=f"gaia-only (N={int(np.sum(m_gaia))})",
        )

    m_both = origins == "both"
    if np.any(m_both):
        ax.scatter(
            star_tbl["x"][m_both],
            star_tbl["y"][m_both],
            s=65,
            c="magenta",
            linewidths=1.4,
            marker="x",
            label=f"both (N={int(np.sum(m_both))})",
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _save_epsf_stamp(epsf_arr: np.ndarray, outpath: Path, title: str, dpi: int = 150):
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(epsf_arr, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _save_patch_background_diagnostics(
    img_raw: np.ndarray,
    img_bsub: np.ndarray,
    bkg_map: np.ndarray,
    outpath: Path,
    title: str,
    dpi: int = 150,
    show_colorbar: bool = True,
):
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)

    def _lims(a: np.ndarray):
        m = np.isfinite(a)
        if not np.any(m):
            return None, None
        return np.percentile(a[m], [5, 99])

    rvmin, rvmax = _lims(img_raw)
    bvmin, bvmax = _lims(bkg_map)
    svmin, svmax = _lims(img_bsub)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    im0 = axes[0].imshow(img_raw, origin="lower", cmap="gray", vmin=rvmin, vmax=rvmax, interpolation="nearest")
    axes[0].set_title("raw patch")
    im1 = axes[1].imshow(bkg_map, origin="lower", cmap="viridis", vmin=bvmin, vmax=bvmax, interpolation="nearest")
    axes[1].set_title("background map")
    im2 = axes[2].imshow(img_bsub, origin="lower", cmap="gray", vmin=svmin, vmax=svmax, interpolation="nearest")
    axes[2].set_title("raw - background")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if bool(show_colorbar):
        fig.colorbar(im0, ax=axes[0], fraction=0.047, pad=0.02)
        fig.colorbar(im1, ax=axes[1], fraction=0.047, pad=0.02)
        fig.colorbar(im2, ax=axes[2], fraction=0.047, pad=0.02)
    fig.suptitle(title, fontsize=11)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _epsf_growth_metrics(epsf_arr: np.ndarray) -> dict:
    arr = np.asarray(epsf_arr, dtype=np.float32)
    ny, nx = arr.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = int(np.ceil(np.nanmax(rr)))
    radii = np.arange(rmax + 1, dtype=np.float32)
    csum = np.array([float(np.sum(arr[rr <= r])) for r in radii], dtype=np.float64)
    total = float(csum[-1]) if csum.size > 0 else float(np.sum(arr))
    if np.isfinite(total) and total != 0.0:
        ee = csum / total
    else:
        ee = np.zeros_like(csum)

    def _r_at(frac: float):
        m = np.where(ee >= float(frac))[0]
        if m.size == 0:
            return np.nan
        return float(radii[m[0]])

    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return dict(
        ee_r50_pix=_r_at(0.5),
        ee_r80_pix=_r_at(0.8),
        ee_r90_pix=_r_at(0.9),
        ee_edge=float(ee[-1]) if ee.size > 0 else np.nan,
        epsf_edge_median=float(np.nanmedian(edge)),
        epsf_edge_mean=float(np.nanmean(edge)),
        epsf_min=float(np.nanmin(arr)),
        epsf_max=float(np.nanmax(arr)),
    )


def _save_epsf_growth_curve(epsf_arr: np.ndarray, outpath: Path, title: str, dpi: int = 150) -> dict:
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(epsf_arr, dtype=np.float32)
    ny, nx = arr.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices(arr.shape, dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = int(np.ceil(np.nanmax(rr)))
    radii = np.arange(rmax + 1, dtype=np.float32)
    csum = np.array([float(np.sum(arr[rr <= r])) for r in radii], dtype=np.float64)
    total = float(csum[-1]) if csum.size > 0 else float(np.sum(arr))
    if np.isfinite(total) and total != 0.0:
        ee = csum / total
    else:
        ee = np.zeros_like(csum)

    qa = _epsf_growth_metrics(arr)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8), constrained_layout=True)
    ax.plot(radii, ee, color="tab:blue", lw=2.0, label="encircled energy")
    ax.axhline(0.5, color="gray", lw=1.0, ls="--")
    ax.axhline(0.8, color="gray", lw=1.0, ls=":")
    if np.isfinite(qa.get("ee_r50_pix", np.nan)):
        ax.axvline(float(qa["ee_r50_pix"]), color="tab:orange", lw=1.2, ls="--", label=f"r50={qa['ee_r50_pix']:.1f}")
    if np.isfinite(qa.get("ee_r80_pix", np.nan)):
        ax.axvline(float(qa["ee_r80_pix"]), color="tab:green", lw=1.2, ls=":", label=f"r80={qa['ee_r80_pix']:.1f}")
    ax.set_xlabel("radius [pix]")
    ax.set_ylabel("encircled energy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2, ls=":")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return qa


def _save_epsf_residual_diagnostics(
    *,
    stars_used,
    epsf_model,
    outpath: Path,
    title: str,
    dpi: int = 150,
    max_stars: int = 25,
) -> dict:
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)
    if stars_used is None or epsf_model is None:
        return dict(resid_nstars=0, resid_rms=np.nan, resid_bias=np.nan, resid_norm_rms=np.nan, resid_norm_bias=np.nan)

    star_list = list(stars_used.all_stars) if hasattr(stars_used, "all_stars") else list(stars_used)
    if len(star_list) == 0:
        return dict(resid_nstars=0, resid_rms=np.nan, resid_bias=np.nan, resid_norm_rms=np.nan, resid_norm_bias=np.nan)

    n = min(int(max_stars), len(star_list))
    data_stack = []
    model_stack = []
    resid_stack = []
    nres_all = []

    for st in star_list[:n]:
        d = np.asarray(st.data, dtype=np.float32)
        try:
            m = np.asarray(st.register_epsf(epsf_model), dtype=np.float32)
        except Exception:
            continue
        if m.shape != d.shape:
            continue
        r = d - m
        data_stack.append(d)
        model_stack.append(m)
        resid_stack.append(r)

        w = np.asarray(getattr(st, "weights", np.ones_like(d)), dtype=np.float32)
        good = np.isfinite(r) & np.isfinite(w) & (w > 0)
        if np.any(good):
            nres = r[good] * np.sqrt(w[good])
            nres_all.append(np.asarray(nres, dtype=np.float32))

    if len(resid_stack) == 0:
        return dict(resid_nstars=0, resid_rms=np.nan, resid_bias=np.nan, resid_norm_rms=np.nan, resid_norm_bias=np.nan)

    dm = np.nanmedian(np.stack(data_stack, axis=0), axis=0)
    mm = np.nanmedian(np.stack(model_stack, axis=0), axis=0)
    rm = np.nanmedian(np.stack(resid_stack, axis=0), axis=0)

    rv = rm[np.isfinite(rm)]
    resid_bias = float(np.nanmedian(rv)) if rv.size > 0 else np.nan
    resid_rms = float(np.sqrt(np.nanmean(rv * rv))) if rv.size > 0 else np.nan

    if len(nres_all) > 0:
        nvec = np.concatenate(nres_all)
        nvec = nvec[np.isfinite(nvec)]
    else:
        nvec = np.array([], dtype=np.float32)
    n_bias = float(np.nanmedian(nvec)) if nvec.size > 0 else np.nan
    n_rms = float(np.sqrt(np.nanmean(nvec * nvec))) if nvec.size > 0 else np.nan

    def _lims(a: np.ndarray):
        m = np.isfinite(a)
        if not np.any(m):
            return None, None
        return np.percentile(a[m], [5, 99])

    dvmin, dvmax = _lims(dm)
    rvlim = np.percentile(np.abs(rv), 99) if rv.size > 0 else 1.0
    if not np.isfinite(rvlim) or rvlim <= 0:
        rvlim = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    axes[0].imshow(dm, origin="lower", cmap="gray", vmin=dvmin, vmax=dvmax, interpolation="nearest")
    axes[0].set_title("median star data")
    axes[1].imshow(mm, origin="lower", cmap="gray", vmin=dvmin, vmax=dvmax, interpolation="nearest")
    axes[1].set_title("registered ePSF model")
    im = axes[2].imshow(rm, origin="lower", cmap="RdBu_r", vmin=-rvlim, vmax=rvlim, interpolation="nearest")
    axes[2].set_title(f"median residual\nrms={resid_rms:.3g}")
    fig.colorbar(im, ax=axes[2], fraction=0.047, pad=0.02)
    axes[3].hist(nvec, bins=60, color="tab:purple", alpha=0.85)
    axes[3].axvline(0.0, color="black", lw=1.0)
    axes[3].set_title(f"normalized residual\nbias={n_bias:.3g} rms={n_rms:.3g}")
    axes[3].set_xlabel("(data-model)/sigma")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)

    return dict(
        resid_nstars=int(len(resid_stack)),
        resid_bias=float(resid_bias),
        resid_rms=float(resid_rms),
        resid_norm_bias=float(n_bias),
        resid_norm_rms=float(n_rms),
    )


def _save_star_local_background_diagnostics(
    *,
    stars_raw,
    stars_bgsub,
    stars_used,
    local_bkg_stats: list[dict],
    outpath: Path,
    title: str,
    dpi: int = 150,
    show_colorbar: bool = True,
) -> dict:
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)

    def _to_list(stars_obj):
        if stars_obj is None:
            return []
        if hasattr(stars_obj, "all_stars"):
            return list(stars_obj.all_stars)
        return list(stars_obj)

    used_list = _to_list(stars_used)
    if len(used_list) == 0:
        used_list = _to_list(stars_bgsub)
    raw_list = _to_list(stars_raw)
    bg_list = _to_list(stars_bgsub)
    if len(used_list) == 0 or len(bg_list) == 0:
        return dict(local_bkg_diag_nstars=0, local_bkg_diag_status="skipped_no_stars")

    def _sid(st):
        v = getattr(st, "id_label", None)
        return "" if v is None else str(v)

    used_ids = [_sid(st) for st in used_list]
    if any(sid == "" for sid in used_ids):
        return dict(
            local_bkg_diag_nstars=0,
            local_bkg_diag_status="skipped_missing_id_label",
            local_bkg_diag_message="Missing id_label detected in used stars.",
        )
    if len(set(used_ids)) != len(used_ids):
        return dict(
            local_bkg_diag_nstars=0,
            local_bkg_diag_status="skipped_duplicate_id_label",
            local_bkg_diag_message="Duplicate id_label detected in used stars.",
        )

    raw_by_id = {_sid(st): st for st in raw_list}
    bg_by_id = {_sid(st): st for st in bg_list}
    stats_by_id = {str(d.get("id_label", "")): d for d in (local_bkg_stats or [])}

    rows = []
    for st in used_list:
        sid = _sid(st)
        st_raw = raw_by_id.get(sid, st)
        st_bg = bg_by_id.get(sid, st)
        stat = stats_by_id.get(sid, {})

        raw = np.asarray(st_raw.data, dtype=np.float32)
        bgsub = np.asarray(st_bg.data, dtype=np.float32)
        w = np.asarray(getattr(st_raw, "weights", np.ones_like(raw)), dtype=np.float32)
        ny, nx = raw.shape

        center = getattr(st_bg, "cutout_center", None)
        if center is None:
            cx = float((nx - 1) / 2.0)
            cy = float((ny - 1) / 2.0)
        else:
            cx = float(center[0])
            cy = float(center[1])

        rin = float(stat.get("annulus_rin", max(0.0, min(nx, ny) * 0.30)))
        rout = float(stat.get("annulus_rout", max(rin + 1.0, min(nx, ny) * 0.45)))
        yy, xx = np.indices(raw.shape, dtype=np.float32)
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ann = (rr2 >= rin * rin) & (rr2 <= rout * rout)
        ann &= np.isfinite(raw) & np.isfinite(w) & (w > 0)

        ann_img = np.full_like(raw, np.nan, dtype=np.float32)
        ann_img[ann] = raw[ann]
        rows.append((sid, raw, ann_img, bgsub, int(np.sum(ann)), float(stat.get("local_bkg", np.nan))))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, max(2.8, 2.35 * n)), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes, dtype=object).reshape(1, 3)

    for i, (sid, raw, ann_img, bgsub, ann_npix, local_bkg) in enumerate(rows):
        m = np.isfinite(raw)
        if np.any(m):
            vmin, vmax = np.percentile(raw[m], [5, 99])
        else:
            vmin, vmax = None, None

        im0 = axes[i, 0].imshow(raw, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        im1 = axes[i, 1].imshow(ann_img, origin="lower", cmap="viridis", interpolation="nearest")
        im2 = axes[i, 2].imshow(bgsub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

        axes[i, 0].set_ylabel(f"id={sid}" if sid != "" else f"star {i}", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("raw stamp", fontsize=10)
            axes[i, 1].set_title("annulus sample", fontsize=10)
            axes[i, 2].set_title("bg-subtracted stamp", fontsize=10)
        axes[i, 1].set_xlabel(f"Nann={ann_npix} bkg={local_bkg:.4g}", fontsize=8)

        for ax in axes[i, :]:
            ax.set_xticks([])
            ax.set_yticks([])
        if bool(show_colorbar):
            fig.colorbar(im0, ax=axes[i, 0], fraction=0.047, pad=0.02)
            fig.colorbar(im1, ax=axes[i, 1], fraction=0.047, pad=0.02)
            fig.colorbar(im2, ax=axes[i, 2], fraction=0.047, pad=0.02)

    fig.suptitle(title, fontsize=11)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return dict(local_bkg_diag_nstars=int(n), local_bkg_diag_status="ok")


def _save_star_montage(
    stars,
    outpath: Path,
    max_stars: int = 25,
    title: str = "stars",
    dpi: int = 150,
    id_to_origin: dict[str, str] | None = None,
):
    import matplotlib.pyplot as plt

    outpath.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(stars, "all_stars"):
        star_list = list(stars.all_stars)
    else:
        star_list = list(stars)

    n = min(int(len(star_list)), int(max_stars))
    if n <= 0:
        return

    color_map = {"gaia": "cyan", "sep": "orange", "both": "magenta", "unknown": "white"}

    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), constrained_layout=True)
    axes = np.array(axes).reshape(nrow, ncol)

    for i in range(nrow * ncol):
        ax = axes.flat[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue

        st = star_list[i]
        cut = np.asarray(st.data, dtype=float)
        m = np.isfinite(cut)
        if np.any(m):
            vmin, vmax = np.percentile(cut[m], [5, 99])
        else:
            vmin, vmax = None, None
        ax.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

        sid = None
        if hasattr(st, "id_label") and st.id_label is not None:
            sid = str(st.id_label)

        origin = "unknown"
        if id_to_origin is not None and sid is not None:
            origin = str(id_to_origin.get(sid, "unknown"))

        col = color_map.get(origin, "white")
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2.0)

        if sid is None:
            ax.set_title(f"{i} | {origin}", fontsize=9)
        else:
            ax.set_title(f"{i} | id={sid} | {origin}", fontsize=9)

    fig.suptitle(title, fontsize=12)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _run_epsf_for_band(
    band: str,
    *,
    progress_queue=None,
) -> list[dict]:
    cfg = _EPSF_CFG
    image_dict = _EPSF_IMAGE_DICT
    gaiaxp = _EPSF_GAIAXP
    use_gaiaxp = _EPSF_USE_GAIAXP
    out_root = _EPSF_OUT_ROOT
    active_patch_tags = _EPSF_ACTIVE_PATCHES

    if cfg is None or image_dict is None or out_root is None:
        raise RuntimeError("EPSF globals not initialized")

    EPSF_NGRID = int(cfg.epsf_ngrid)

    d = image_dict[band]
    img_full = np.asarray(d["img_scaled"], dtype=np.float32)
    sig_full = np.asarray(d["sigma_sky_scaled"], dtype=np.float32)
    bad_full = np.asarray(d.get("bad", ~np.isfinite(img_full)), dtype=bool)

    saturate_scaled = None
    try:
        sat = float(d.get("hdr", {}).get("SATURATE", np.inf))
        if np.isfinite(sat):
            saturate_scaled = sat * float(d.get("scale", 1.0))
    except Exception:
        saturate_scaled = None

    invvar_full = np.zeros_like(img_full, dtype=np.float32)
    good = (~bad_full) & np.isfinite(sig_full) & (sig_full > 0)
    invvar_full[good] = 1.0 / np.maximum(sig_full[good] * sig_full[good], 1e-12)

    gx = gy = None
    mode_raw = str(getattr(cfg, "psfstar_mode", "sep+gaia")).strip().lower()
    mode = mode_raw
    if mode not in ("sep", "sep+gaia", "gaia"):
        logger.warning("%s: psfstar_mode=%r invalid; using 'sep+gaia'.", band, mode_raw)
        mode = "sep+gaia"

    forced_r_pix = None
    has_peeing = False
    has_seeing = False
    try:
        hdr = d.get("hdr", {})
        if hdr is not None:
            if ("PEEING" in hdr) and np.isfinite(float(hdr["PEEING"])):
                fwhm_pix = float(hdr["PEEING"])
                has_peeing = True
            elif ("SEEING" in hdr) and np.isfinite(float(hdr["SEEING"])):
                seeing_arcsec = float(hdr["SEEING"])
                has_seeing = True
                pixscales = proj_plane_pixel_scales(d["wcs"]) * 3600.0
                pixscale = float(np.nanmedian(pixscales))
                fwhm_pix = seeing_arcsec / pixscale if pixscale > 0 else np.nan
            else:
                fwhm_pix = np.nan

            if np.isfinite(fwhm_pix) and fwhm_pix > 0:
                k = float(getattr(cfg, "gaia_forced_k_fwhm", 1.0))
                rmin = float(getattr(cfg, "gaia_forced_rmin_pix", 2.0))
                rmax = float(getattr(cfg, "gaia_forced_rmax_pix", 8.0))
                r = k * fwhm_pix
                r = max(rmin, min(rmax, r))
                r = min(r, float(int(cfg.cutout_size // 2 - 2)))
                forced_r_pix = float(r)
    except Exception:
        forced_r_pix = None

    if forced_r_pix is None:
        if has_peeing:
            src = "PEEING (invalid)"
        elif has_seeing:
            src = "SEEING (invalid)"
        else:
            src = "PEEING/SEEING missing"
        logger.warning("%s: %s; using default forced radius.", band, src)

    if use_gaiaxp and (mode in ("sep+gaia", "gaia")) and (gaiaxp is not None):
        mag_col = f"mag_{band}"
        if mag_col in gaiaxp.columns:
            g = gaiaxp[["ra", "dec", mag_col]].copy()
            g = g[np.isfinite(g["ra"]) & np.isfinite(g["dec"]) & np.isfinite(g[mag_col])]
            g = g[(g[mag_col] >= float(cfg.gaia_mag_min)) & (g[mag_col] <= float(cfg.gaia_mag_max))]
            if len(g) > 0:
                xpix, ypix = d["wcs"].all_world2pix(g["ra"].to_numpy(float), g["dec"].to_numpy(float), 0)
                gx = np.asarray(xpix, dtype=float)
                gy = np.asarray(ypix, dtype=float)
                mfin = np.isfinite(gx) & np.isfinite(gy)
                gx = gx[mfin]
                gy = gy[mfin]

    band_dir = out_root / str(band)
    band_dir.mkdir(parents=True, exist_ok=True)

    ny, nx = img_full.shape
    out: list[dict] = []
    n_total_patch = (
        int(len(active_patch_tags))
        if active_patch_tags is not None
        else int(EPSF_NGRID * EPSF_NGRID)
    )
    n_done_patch = 0
    n_ok_patch = 0
    n_fail_patch = 0
    progress_every = max(1, n_total_patch // 10)
    t0 = time.time()
    if n_total_patch == 0:
        logger.warning("[epsf %s] no active ePSF cells; skipping band", band)
        return out

    def _emit_band_progress(*, force_log: bool = False) -> None:
        if not force_log and (n_done_patch % progress_every) != 0:
            return
        def _format_eta(seconds: float) -> str:
            if not (seconds >= 0) or seconds == float("inf"):
                return "--:--"
            s = int(round(seconds))
            m, ss = divmod(s, 60)
            h, mm = divmod(m, 60)
            if h > 0:
                return f"{h:d}:{mm:02d}:{ss:02d}"
            return f"{mm:02d}:{ss:02d}"

        if progress_queue is not None:
            progress_queue.put(
                dict(
                    band=str(band),
                    done=int(n_done_patch),
                    total=int(n_total_patch),
                    ok=int(n_ok_patch),
                    fail=int(n_fail_patch),
                    elapsed=float(max(1e-6, time.time() - t0)),
                    final=bool(n_done_patch >= n_total_patch),
                )
            )
            return
        rate = n_done_patch / max(1e-6, time.time() - t0)
        remain = max(0, n_total_patch - n_done_patch)
        eta = float("inf") if rate <= 0 else remain / rate
        logger.info(
            "[epsf %s] %d/%d patches ok=%d fail=%d rate=%.2f/s eta=%s",
            band,
            n_done_patch,
            n_total_patch,
            n_ok_patch,
            n_fail_patch,
            rate,
            _format_eta(eta),
        )

    need = int(cfg.cutout_size // 2 + cfg.edge_pad)

    cand_x_full = cand_y_full = cand_score_full = cand_q_full = None
    if not bool(getattr(cfg, "background_subtract_patch", True)):
        cand_full = _sep_extract_candidates_full(
            img_full=img_full,
            sigma_full=sig_full,
            bad_full=bad_full,
            cfg=cfg,
            saturate=saturate_scaled,
        )
        if cand_full is not None:
            cand_x_full, cand_y_full, cand_score_full, cand_q_full = cand_full

    for pr in range(EPSF_NGRID):
        for pc in range(EPSF_NGRID):
            x0, x1, y0, y1 = _grid_bounds(nx, ny, EPSF_NGRID, pr, pc)

            img = img_full[y0:y1, x0:x1]
            sig = sig_full[y0:y1, x0:x1]
            bad = bad_full[y0:y1, x0:x1]
            inv = invvar_full[y0:y1, x0:x1]
            img_sel = img
            bkg_map = np.zeros_like(img, dtype=np.float32)
            bkg_rms = np.zeros_like(img, dtype=np.float32)
            patch_bkg_stats = dict(
                patch_bkg_median=0.0,
                patch_bkg_mean=0.0,
                patch_bkg_rms_median=np.nan,
                patch_bkg_rms_mean=np.nan,
                patch_bkg_mask_frac=float(np.mean(bad)),
                patch_bkg_status="disabled",
            )

            patch_tag = f"r{pr:02d}_c{pc:02d}"
            if active_patch_tags is not None and patch_tag not in active_patch_tags:
                continue
            patch_dir = band_dir / patch_tag
            patch_dir.mkdir(parents=True, exist_ok=True)

            if bool(getattr(cfg, "background_subtract_patch", True)):
                img_sel, bkg_map, bkg_rms, patch_bkg_stats = _estimate_patch_background(
                    img=img,
                    sigma=sig,
                    bad=bad,
                    cfg=cfg,
                )
                if bool(getattr(cfg, "save_patch_background_diagnostics", True)):
                    _save_patch_background_diagnostics(
                        img_raw=img,
                        img_bsub=img_sel,
                        bkg_map=bkg_map,
                        outpath=patch_dir / "background_diagnostics.png",
                        title=f"{band} {patch_tag} | patch local background",
                        dpi=_EPSF_PLOT_DPI,
                        show_colorbar=bool(getattr(cfg, "diagnostics_show_colorbar", True)),
                    )

            seed_tbl = None
            if (mode in ("sep+gaia", "gaia")) and (gx is not None) and (gy is not None) and (len(gx) > 0):
                m = (gx >= x0 + need) & (gx < x1 - need - 1) & (gy >= y0 + need) & (gy < y1 - need - 1)
                if np.any(m):
                    seed_tbl = Table(
                        data=[(gx[m] - x0).astype(np.float32), (gy[m] - y0).astype(np.float32)],
                        names=("x", "y"),
                    )

            if cand_x_full is not None:
                msep = (
                    (cand_x_full >= x0 + need)
                    & (cand_x_full < x1 - need - 1)
                    & (cand_y_full >= y0 + need)
                    & (cand_y_full < y1 - need - 1)
                )
                xs = (cand_x_full[msep] - float(x0)).astype(float)
                ys = (cand_y_full[msep] - float(y0)).astype(float)
                ss = cand_score_full[msep].astype(float)
                qq = cand_q_full[msep].astype(float)

                cand = list(zip(xs.tolist(), ys.tolist(), ss.tolist()))
                cand_xy = list(zip(xs.tolist(), ys.tolist()))
                cand_q = qq.tolist()

                star_tbl = _select_psf_stars_from_candidates(
                    img=img_sel,
                    sigma=sig,
                    bad=bad,
                    cfg=cfg,
                    cand=cand,
                    cand_xy=cand_xy,
                    cand_q=cand_q,
                    seed_tbl=seed_tbl,
                    saturate=saturate_scaled,
                    mode=mode,
                    forced_r_pix=forced_r_pix,
                )
            else:
                star_tbl = _select_psf_stars_sep(
                    img=img_sel,
                    sigma=sig,
                    bad=bad,
                    cfg=cfg,
                    seed_tbl=seed_tbl,
                    saturate=saturate_scaled,
                    mode=mode,
                    forced_r_pix=forced_r_pix,
                )

            n_seed = int(len(seed_tbl)) if seed_tbl is not None else 0
            _save_star_overlay(
                img,
                star_tbl,
                outpath=patch_dir / "psfstars.png",
                title=f"{band} {patch_tag} | stars selected={len(star_tbl)} (gaia_seed={n_seed})",
                dpi=_EPSF_PLOT_DPI,
            )
            if bool(getattr(cfg, "background_subtract_patch", True)):
                _save_star_overlay(
                    img_sel,
                    star_tbl,
                    outpath=patch_dir / "psfstars_bgsub.png",
                    title=f"{band} {patch_tag} | stars selected on bg-sub image",
                    dpi=_EPSF_PLOT_DPI,
                )

            res = _build_epsf_from_invvar(img=img_sel, invvar=inv, star_tbl=star_tbl, cfg=cfg)
            if res is None:
                meta = dict(
                    band=str(band),
                    patch=patch_tag,
                    x0=int(x0),
                    x1=int(x1),
                    y0=int(y0),
                    y1=int(y1),
                    epsf_ngrid=int(EPSF_NGRID),
                    psfstar_mode=str(mode),
                    gaia_snap_to_sep=bool(cfg.gaia_snap_to_sep),
                    gaia_mag_min=float(cfg.gaia_mag_min),
                    gaia_mag_max=float(cfg.gaia_mag_max),
                    nstars_seed=int(n_seed),
                    nstars_sel=int(len(star_tbl)),
                    nstars_extract=0,
                    nstars_used=0,
                    epsf_shape=None,
                    status="fail",
                    error="None returned",
                    patch_background=patch_bkg_stats,
                )
                (patch_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                out.append(meta)
                n_done_patch += 1
                n_fail_patch += 1
                _emit_band_progress(force_log=False)
                continue

            epsf_arr = res.get("epsf_arr", None)
            n_extract = int(res.get("nstars_extract", 0))
            n_used = int(res.get("nstars_used", 0))
            err_msg = res.get("error", None)
            local_bkg_stats = res.get("local_bkg_stats", []) or []
            local_bkg_vals = [float(r.get("local_bkg")) for r in local_bkg_stats if np.isfinite(float(r.get("local_bkg", np.nan)))]
            local_bkg_ok = int(sum(1 for r in local_bkg_stats if str(r.get("status", "")) == "ok"))
            local_bkg_summary = dict(
                local_bkg_nstars=int(len(local_bkg_stats)),
                local_bkg_ok_nstars=int(local_bkg_ok),
                local_bkg_median=float(np.nanmedian(local_bkg_vals)) if len(local_bkg_vals) > 0 else np.nan,
                local_bkg_mean=float(np.nanmean(local_bkg_vals)) if len(local_bkg_vals) > 0 else np.nan,
            )

            if epsf_arr is None or n_used <= 0:
                meta = dict(
                    band=str(band),
                    patch=patch_tag,
                    x0=int(x0),
                    x1=int(x1),
                    y0=int(y0),
                    y1=int(y1),
                    epsf_ngrid=int(EPSF_NGRID),
                    psfstar_mode=str(mode),
                    gaia_snap_to_sep=bool(cfg.gaia_snap_to_sep),
                    gaia_mag_min=float(cfg.gaia_mag_min),
                    gaia_mag_max=float(cfg.gaia_mag_max),
                    nstars_seed=int(n_seed),
                    nstars_sel=int(len(star_tbl)),
                    nstars_extract=int(n_extract),
                    nstars_used=0,
                    epsf_shape=None,
                    status="fail",
                    error=err_msg,
                    patch_background=patch_bkg_stats,
                    local_star_background=local_bkg_summary,
                )
                (patch_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                out.append(meta)
                n_done_patch += 1
                n_fail_patch += 1
                _emit_band_progress(force_log=False)
                continue

            np.save(patch_dir / "epsf.npy", epsf_arr)
            _save_epsf_stamp(epsf_arr, outpath=patch_dir / "epsf.png", title=f"{band} {patch_tag} | used={n_used}/{n_extract}", dpi=_EPSF_PLOT_DPI)
            epsf_qa = dict(res.get("epsf_qa", {}))
            if bool(getattr(cfg, "save_growth_curve", True)):
                epsf_qa.update(
                    _save_epsf_growth_curve(
                        epsf_arr,
                        outpath=patch_dir / "epsf_growth_curve.png",
                        title=f"{band} {patch_tag} | growth curve",
                        dpi=_EPSF_PLOT_DPI,
                    )
                )
            residual_qa = dict(resid_nstars=0, resid_rms=np.nan, resid_bias=np.nan, resid_norm_rms=np.nan, resid_norm_bias=np.nan)
            if bool(getattr(cfg, "save_residual_diagnostics", True)):
                residual_qa = _save_epsf_residual_diagnostics(
                    stars_used=res.get("stars_used"),
                    epsf_model=res.get("epsf_model"),
                    outpath=patch_dir / "epsf_residual_diagnostics.png",
                    title=f"{band} {patch_tag} | star - ePSF residual diagnostics",
                    dpi=_EPSF_PLOT_DPI,
                    max_stars=int(getattr(cfg, "residual_diag_max_stars", 25)),
                )
            local_bkg_diag = dict(local_bkg_diag_nstars=0)
            if (
                bool(getattr(cfg, "star_local_bkg_subtract", True))
                and bool(getattr(cfg, "save_star_local_background_diagnostics", True))
                and (len(local_bkg_stats) > 0)
            ):
                local_bkg_diag = _save_star_local_background_diagnostics(
                    stars_raw=res.get("stars_all_raw"),
                    stars_bgsub=res.get("stars_all_bgsub"),
                    stars_used=res.get("stars_used"),
                    local_bkg_stats=local_bkg_stats,
                    outpath=patch_dir / "star_local_background_diagnostics.png",
                    title=f"{band} {patch_tag} | per-star annulus background diagnostics",
                    dpi=_EPSF_PLOT_DPI,
                    show_colorbar=bool(getattr(cfg, "diagnostics_show_colorbar", True)),
                )
                if str(local_bkg_diag.get("local_bkg_diag_status", "")) != "ok":
                    logger.warning(
                        "%s %s: skipped star_local_background_diagnostics (%s)",
                        band,
                        patch_tag,
                        local_bkg_diag.get("local_bkg_diag_status", "unknown"),
                    )

            id_to_origin = res.get("id_to_origin", None)
            _save_star_montage(
                res["stars_used"],
                outpath=patch_dir / "used_stars.png",
                max_stars=cfg.save_star_montage_max,
                title=f"{band} {patch_tag} | fitted stars (first {cfg.save_star_montage_max})",
                id_to_origin=id_to_origin,
                dpi=_EPSF_PLOT_DPI,
            )
            _save_star_montage(
                res["stars_all"],
                outpath=patch_dir / "extracted_stars.png",
                max_stars=cfg.save_star_montage_max,
                title=f"{band} {patch_tag} | extracted stars (first {cfg.save_star_montage_max})",
                id_to_origin=id_to_origin,
                dpi=_EPSF_PLOT_DPI,
            )

            meta = dict(
                band=str(band),
                patch=patch_tag,
                x0=int(x0),
                x1=int(x1),
                y0=int(y0),
                y1=int(y1),
                epsf_ngrid=int(EPSF_NGRID),
                psfstar_mode=str(mode),
                gaia_snap_to_sep=bool(cfg.gaia_snap_to_sep),
                gaia_mag_min=float(cfg.gaia_mag_min),
                gaia_mag_max=float(cfg.gaia_mag_max),
                nstars_seed=int(n_seed),
                nstars_sel=int(len(star_tbl)),
                nstars_extract=int(n_extract),
                nstars_used=int(n_used),
                nstars_fitfail=int(max(0, n_extract - n_used)),
                epsf_shape=list(map(int, epsf_arr.shape)),
                patch_background=patch_bkg_stats,
                local_star_background=local_bkg_summary,
                local_bkg_diagnostics=local_bkg_diag,
                epsf_qa=epsf_qa,
                residual_qa=residual_qa,
                cfg=dict(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}),
                status="ok",
            )
            (patch_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            out.append(meta)
            n_done_patch += 1
            n_ok_patch += 1
            _emit_band_progress(force_log=False)

    _emit_band_progress(force_log=True)
    return out


def build_epsf(
    *,
    cfg: EPSFConfig,
    outputs_dir: Path,
    image_dict: dict,
    gaiaxp,
    use_gaiaxp: bool,
    parallel_bands: bool,
    max_workers: int | None,
    active_epsf_tags: set[str] | None = None,
    plot_dpi: int = 150,
) -> Path:
    EPSF_NGRID = int(cfg.epsf_ngrid)

    out_root = outputs_dir
    out_root.mkdir(parents=True, exist_ok=True)

    bands = list(image_dict.keys())
    logger.info("Bands: %s", bands)
    logger.info("Image shape: %s", next(iter(image_dict.values()))["img_scaled"].shape)

    if max_workers is None:
        MAX_WORKERS = min(len(bands), int(os.cpu_count() or 1))
    else:
        MAX_WORKERS = int(max_workers)
    logger.info("MAX_WORKERS: %d", MAX_WORKERS)
    # For parallel bands in an interactive terminal, render one live line per worker.
    use_inline_progress = sys.stderr.isatty() and (not bool(parallel_bands))
    use_worker_lines = sys.stderr.isatty() and bool(parallel_bands)
    _inline_width = max(40, shutil.get_terminal_size((80, 24)).columns - 1)

    global _EPSF_IMAGE_DICT, _EPSF_CFG, _EPSF_GAIAXP, _EPSF_USE_GAIAXP, _EPSF_OUT_ROOT, _EPSF_ACTIVE_PATCHES, _EPSF_PLOT_DPI
    _EPSF_IMAGE_DICT = image_dict
    _EPSF_CFG = cfg
    _EPSF_PLOT_DPI = int(plot_dpi)
    _EPSF_GAIAXP = gaiaxp
    _EPSF_USE_GAIAXP = bool(use_gaiaxp)
    _EPSF_OUT_ROOT = out_root
    _EPSF_ACTIVE_PATCHES = set(active_epsf_tags) if active_epsf_tags is not None else None
    active_patch_total = (
        len(_EPSF_ACTIVE_PATCHES)
        if _EPSF_ACTIVE_PATCHES is not None
        else int(EPSF_NGRID * EPSF_NGRID)
    )
    if _EPSF_ACTIVE_PATCHES is not None:
        logger.info("Active ePSF cells: %d/%d", active_patch_total, EPSF_NGRID * EPSF_NGRID)

    summary: list[dict] = []
    tstart = time.time()
    n_total = len(bands)
    band_done = 0
    band_fail = 0
    patch_ok = 0
    patch_fail = 0
    last_non_tty_log = 0.0

    def _format_eta(seconds: float) -> str:
        if not (seconds >= 0) or seconds == float("inf"):
            return "--:--"
        s = int(round(seconds))
        m, ss = divmod(s, 60)
        h, mm = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def _progress_bar(done: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return "-" * width
        frac = max(0.0, min(1.0, float(done) / float(total)))
        filled = int(round(frac * width))
        return "#" * filled + "-" * (width - filled)

    def _emit_progress(*, running: int, force_log: bool = False) -> None:
        nonlocal last_non_tty_log
        elapsed = max(1e-6, time.time() - tstart)
        rate = band_done / elapsed
        remaining = max(0, n_total - band_done)
        eta = float("inf") if rate <= 0 else remaining / rate
        msg = (
            f"ePSF bands [{_progress_bar(band_done, n_total)}] "
            f"{band_done}/{n_total} running={running} band_fail={band_fail} "
            f"patch_ok={patch_ok} patch_fail={patch_fail} eta={_format_eta(eta)}"
        )
        if use_inline_progress:
            sys.stderr.write("\r" + msg[:_inline_width].ljust(_inline_width))
            sys.stderr.flush()
        else:
            now = time.time()
            if force_log or (now - last_non_tty_log) >= 15.0:
                logger.info(msg)
                last_non_tty_log = now

    def _handle_band_result(band_name: str, res: list[dict]) -> None:
        nonlocal band_done, patch_ok, patch_fail
        summary.extend(res)
        ok = sum(1 for r in res if r.get("status") == "ok")
        fail = len(res) - ok
        patch_ok += ok
        patch_fail += fail
        band_done += 1
        logger.debug("[band done] %s: ok=%d fail=%d", band_name, ok, fail)

    def _run_parallel_with_live_worker_lines(*, ex, submit_fn) -> None:
        nonlocal band_done, band_fail
        if not bands:
            return

        n_slots = max(1, min(int(MAX_WORKERS), len(bands)))
        manager = mp.Manager()
        progress_q = manager.Queue()
        remaining = list(bands)
        active: dict = {}
        slot_to_band: dict[int, str | None] = {i: None for i in range(n_slots)}
        slot_notice: dict[int, str] = {i: "" for i in range(n_slots)}
        band_state: dict[str, dict[str, float | int | bool]] = {}
        active_slots: list[int] = list(range(n_slots))

        for slot in active_slots:
            if not remaining:
                break
            b = remaining.pop(0)
            fut = submit_fn(ex, b, progress_q)
            active[fut] = (str(b), slot)
            slot_to_band[slot] = str(b)
            slot_notice[slot] = ""
            band_state[str(b)] = dict(done=0, total=int(active_patch_total), ok=0, fail=0, elapsed=0.0)

        # Reserve terminal lines for worker bars.
        sys.stderr.write("\n" * n_slots)
        sys.stderr.write(f"\x1b[{n_slots}A")
        sys.stderr.flush()
        first_render = True

        def _bar(done: int, total: int, width: int = 20) -> str:
            if total <= 0:
                return "-" * width
            frac = max(0.0, min(1.0, float(done) / float(total)))
            fill = int(round(frac * width))
            return "#" * fill + "-" * (width - fill)

        def _format_eta(seconds: float) -> str:
            if not (seconds >= 0) or seconds == float("inf"):
                return "--:--"
            s = int(round(seconds))
            m, ss = divmod(s, 60)
            h, mm = divmod(m, 60)
            if h > 0:
                return f"{h:d}:{mm:02d}:{ss:02d}"
            return f"{mm:02d}:{ss:02d}"

        def _render_worker_lines() -> None:
            nonlocal first_render
            if first_render:
                first_render = False
            else:
                sys.stderr.write(f"\x1b[{n_slots}A")
            for slot in range(n_slots):
                b = slot_to_band.get(slot)
                if b is None:
                    line = f"[w{slot + 1:02d}] idle"
                else:
                    st = band_state.get(b, {})
                    done_i = int(st.get("done", 0))
                    total_i = int(st.get("total", int(EPSF_NGRID * EPSF_NGRID)))
                    ok_i = int(st.get("ok", 0))
                    fail_i = int(st.get("fail", 0))
                    elapsed_i = float(st.get("elapsed", 0.0))
                    rate_i = done_i / max(1e-6, elapsed_i)
                    remain_i = max(0, total_i - done_i)
                    eta_i = float("inf") if rate_i <= 0 else remain_i / rate_i
                    pct = int((100.0 * done_i) / max(1, total_i))
                    line = (
                        f"[w{slot + 1:02d} {b}] [{_bar(done_i, total_i)}] "
                        f"{pct:3d}% {done_i}/{total_i} ok={ok_i} fail={fail_i} "
                        f"rate={rate_i:.2f}/s eta={_format_eta(eta_i)}"
                    )
                notice = slot_notice.get(slot, "")
                if notice:
                    line = f"{line} | {notice}"
                sys.stderr.write("\r" + line[:_inline_width].ljust(_inline_width) + "\n")
            sys.stderr.flush()

        _render_worker_lines()
        while active:
            # Drain worker progress messages.
            while True:
                try:
                    msg = progress_q.get_nowait()
                except queue_mod.Empty:
                    break
                b = str(msg.get("band"))
                st = band_state.setdefault(b, {})
                st["done"] = int(msg.get("done", st.get("done", 0)))
                st["total"] = int(msg.get("total", st.get("total", int(active_patch_total))))
                st["ok"] = int(msg.get("ok", st.get("ok", 0)))
                st["fail"] = int(msg.get("fail", st.get("fail", 0)))
                st["elapsed"] = float(msg.get("elapsed", st.get("elapsed", 0.0)))
            _render_worker_lines()

            done_now, _ = cf.wait(set(active.keys()), timeout=0.5, return_when=cf.FIRST_COMPLETED)
            if not done_now:
                continue
            for fut in done_now:
                b, slot = active.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.error("[band FAIL] %s: %s", b, msg)
                    summary.append(dict(band=str(b), status="band_fail", error=msg))
                    band_done += 1
                    band_fail += 1
                    slot_notice[slot] = f"done {b} (worker error)"
                else:
                    done_total = int(len(res))
                    done_ok = int(sum(1 for r in res if r.get("status") == "ok"))
                    done_fail = int(done_total - done_ok)
                    _handle_band_result(str(b), res)
                    slot_notice[slot] = f"done {b} ({done_ok}/{done_total} ok, fail={done_fail})"
                # Assign next band to this freed slot.
                if remaining:
                    nb = remaining.pop(0)
                    nfut = submit_fn(ex, nb, progress_q)
                    active[nfut] = (str(nb), slot)
                    slot_to_band[slot] = str(nb)
                    slot_notice[slot] = f"{slot_notice[slot]} -> start {nb}"
                    band_state[str(nb)] = dict(done=0, total=int(active_patch_total), ok=0, fail=0, elapsed=0.0)
                else:
                    slot_to_band[slot] = None
            _render_worker_lines()

        # Leave the cursor below the worker lines.
        sys.stderr.flush()
        manager.shutdown()

    if parallel_bands:
        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = None
        if use_worker_lines:
            if ctx is not None:
                with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
                    _run_parallel_with_live_worker_lines(
                        ex=ex,
                        submit_fn=lambda pool, b, q: pool.submit(_run_epsf_for_band, b, progress_queue=q),
                    )
            else:
                with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    _run_parallel_with_live_worker_lines(
                        ex=ex,
                        submit_fn=lambda pool, b, q: pool.submit(_run_epsf_for_band, b, progress_queue=q),
                    )
        elif ctx is not None:
            with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
                futs = {ex.submit(_run_epsf_for_band, b): b for b in bands}
                pending = set(futs.keys())
                _emit_progress(running=len(pending), force_log=True)
                while pending:
                    done_now, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
                    if not done_now:
                        _emit_progress(running=len(pending), force_log=False)
                        continue
                    for fut in done_now:
                        b = futs[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            msg = f"{type(e).__name__}: {e}"
                            logger.error("[band FAIL] %s: %s", b, msg)
                            summary.append(dict(band=str(b), status="band_fail", error=msg))
                            band_done += 1
                            band_fail += 1
                            continue
                        _handle_band_result(str(b), res)
                    _emit_progress(running=len(pending), force_log=True)
        else:
            with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = {ex.submit(_run_epsf_for_band, b): b for b in bands}
                pending = set(futs.keys())
                _emit_progress(running=len(pending), force_log=True)
                while pending:
                    done_now, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
                    if not done_now:
                        _emit_progress(running=len(pending), force_log=False)
                        continue
                    for fut in done_now:
                        b = futs[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            msg = f"{type(e).__name__}: {e}"
                            logger.error("[band FAIL] %s: %s", b, msg)
                            summary.append(dict(band=str(b), status="band_fail", error=msg))
                            band_done += 1
                            band_fail += 1
                            continue
                        _handle_band_result(str(b), res)
                    _emit_progress(running=len(pending), force_log=True)
    else:
        _emit_progress(running=1 if n_total > 0 else 0, force_log=True)
        for b in bands:
            res = _run_epsf_for_band(b)
            _handle_band_result(str(b), res)
            _emit_progress(running=max(0, n_total - band_done), force_log=True)

    if use_inline_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    summary_path = out_root / "epsf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary: %s", summary_path)
    logger.info("Total patches: %d", len(summary))
    logger.info(
        "OK: %d FAIL: %d",
        sum(1 for s in summary if s.get("status") == "ok"),
        sum(1 for s in summary if s.get("status") != "ok"),
    )
    return summary_path
