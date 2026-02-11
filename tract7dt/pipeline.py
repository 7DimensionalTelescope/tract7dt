from __future__ import annotations

import concurrent.futures as cf
import logging
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval

from .epsf import EPSFConfig, build_epsf as _build_epsf_impl
from .merge import merge_catalogs
from .patch_inputs import build_patch_inputs as _build_patch_inputs_impl
from .patches import build_patches as _build_patches_impl
from .run_patches import run_subprocesses

logger = logging.getLogger("tract7dt.pipeline")


def _log(msg: str) -> None:
    text = str(msg).strip()
    if text.startswith("[WARN]"):
        logger.warning(text[len("[WARN]") :].strip())
    elif text.startswith("[CHECK]") and "FAILED" in text:
        logger.warning(text)
    else:
        logger.info(text)


def _format_elapsed(dt: float) -> str:
    return f"{dt:.2f}s"


def _summarize_fit_results(final_catalog: Path) -> None:
    if not final_catalog.exists():
        _log(f"[WARN] Final catalog not found for summary: {final_catalog}")
        return
    try:
        df = pd.read_csv(final_catalog)
    except Exception as e:
        _log(f"[WARN] Could not read final catalog for summary: {e}")
        return
    if "opt_converged" not in df.columns or "opt_hit_max_iters" not in df.columns:
        _log("[WARN] Summary skipped: opt_converged/opt_hit_max_iters missing.")
        return
    conv = pd.to_numeric(df["opt_converged"], errors="coerce").fillna(0).astype(int).sum()
    hit = pd.to_numeric(df["opt_hit_max_iters"], errors="coerce").fillna(0).astype(int).sum()
    _log(f"Fit summary: converged={conv}, hit_max_iters={hit}, total={len(df)}")


def _read_image_list(list_path: Path, config_dir: Path) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Image list file not found: {list_path}")
    paths: list[Path] = []
    for line in list_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = config_dir / p
        paths.append(p.resolve())
    return paths


def _wcs_close(w0: WCS, w1: WCS, tol: dict[str, float]) -> bool:
    a = w0.wcs
    b = w1.wcs

    def _allclose(x, y, key):
        t = float(tol.get(key, 0.0))
        return np.allclose(np.asarray(x), np.asarray(y), rtol=0.0, atol=t)

    if not _allclose(a.crval, b.crval, "crval"):
        return False
    if not _allclose(a.crpix, b.crpix, "crpix"):
        return False

    if a.has_cd() and b.has_cd():
        if not _allclose(a.cd, b.cd, "cd"):
            return False
    elif (not a.has_cd()) and (not b.has_cd()):
        if not _allclose(a.cdelt, b.cdelt, "cdelt"):
            return False
        if not _allclose(a.pc, b.pc, "cd"):
            return False
    else:
        return False

    if tuple(a.ctype) != tuple(b.ctype):
        return False
    return True


def _validate_headers(hdr, fp: Path) -> None:
    req = ["FILTER", "ZP_AUTO", "SKYSIG", "EGAIN"]
    for k in req:
        if k not in hdr:
            raise RuntimeError(f"Missing {k} in {fp}")
    if not str(hdr.get("FILTER", "")).strip():
        raise RuntimeError(f"Empty FILTER in {fp}")


def _prepare_image_frame(fp: Path, zp_ref: float) -> dict:
    with fits.open(fp, memmap=True) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header

    _validate_headers(hdr, fp)

    filt = hdr.get("FILTER", "UNKNOWN").strip()
    zp = hdr.get("ZP_AUTO")
    if zp is None:
        raise RuntimeError(f"Missing ZP_AUTO in {fp}")

    skysig = hdr.get("SKYSIG")
    if skysig is None or (not np.isfinite(skysig)) or skysig <= 0:
        raise RuntimeError(f"Bad/Missing SKYSIG in {fp}")

    egain = hdr.get("EGAIN")
    if egain is None or (not np.isfinite(egain)) or egain <= 0:
        raise RuntimeError(f"Bad/Missing EGAIN (e-/ADU) in {fp}")

    warn_messages: list[str] = []
    if "SATURATE" not in hdr:
        warn_messages.append(f"[WARN] SATURATE missing in {fp}; using inf.")
        saturate = np.inf
    else:
        saturate = hdr.get("SATURATE", np.inf)
        try:
            if not np.isfinite(float(saturate)):
                warn_messages.append(f"[WARN] SATURATE invalid in {fp}; using inf.")
                saturate = np.inf
        except Exception:
            warn_messages.append(f"[WARN] SATURATE invalid in {fp}; using inf.")
            saturate = np.inf

    scale = 10.0 ** (-0.4 * (float(zp) - zp_ref))
    img_scaled = img * scale

    bad = ~np.isfinite(img)
    satur_mask = np.zeros_like(bad, dtype=bool)
    if np.isfinite(saturate):
        satur_mask = np.isfinite(img) & (img >= float(saturate / 2))
        bad |= satur_mask

    sigma_sky_adu = float(skysig)
    sigma_sky_scaled = np.full_like(img, np.inf, dtype=np.float32)
    sigma_sky_scaled[~bad] = sigma_sky_adu * scale

    var_src_adu = np.zeros_like(img, dtype=np.float32)
    good = ~bad
    var_src_adu[good] = np.maximum(img[good], 0.0) / float(egain)

    var_total_adu = (sigma_sky_adu * sigma_sky_adu) + var_src_adu
    sigma_total_scaled = np.full_like(img, np.inf, dtype=np.float32)
    sigma_total_scaled[good] = np.sqrt(var_total_adu[good]) * scale

    wcs = WCS(hdr)
    return dict(
        filt=filt,
        shape=img.shape,
        wcs=wcs,
        img_scaled=img_scaled,
        sigma_total_scaled=sigma_total_scaled,
        sigma_sky_scaled=sigma_sky_scaled,
        bad=bad,
        satur_mask=satur_mask,
        hdr=hdr,
        scale=scale,
        fp=str(fp),
        warn_messages=warn_messages,
    )


def _accumulate_white_chunk(
    image_items: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    y0: int,
    y1: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    first_img = image_items[0][0]
    w = first_img.shape[1]
    h = y1 - y0
    white_num = np.zeros((h, w), dtype=np.float32)
    white_den = np.zeros((h, w), dtype=np.float32)
    white_mask = np.zeros((h, w), dtype=bool)

    for img_s, sig_sky, bad in image_items:
        img_c = np.asarray(img_s[y0:y1, :], dtype=np.float32)
        sig_c = np.asarray(sig_sky[y0:y1, :], dtype=np.float32)
        bad_c = np.asarray(bad[y0:y1, :], dtype=bool)

        wt = np.zeros_like(img_c, dtype=np.float32)
        good = (~bad_c) & np.isfinite(sig_c) & (sig_c > 0)
        wt[good] = 1.0 / np.maximum(sig_c[good] * sig_c[good], 1e-12)

        white_num += wt * img_c
        white_den += wt
        white_mask |= bad_c

    return y0, y1, white_num, white_den, white_mask


def _disk_offsets(radius_pix: float) -> list[tuple[int, int]]:
    r = max(0, int(np.ceil(float(radius_pix))))
    offs: list[tuple[int, int]] = []
    rr2 = float(radius_pix) * float(radius_pix)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx * dx + dy * dy) <= rr2:
                offs.append((dy, dx))
    if not offs:
        offs.append((0, 0))
    return offs


def _filter_sources_near_saturation(
    *,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, dict],
    ra_col: str,
    dec_col: str,
    radius_pix: float,
    require_all_bands: bool,
    return_removed_mask: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    if len(input_catalog) == 0:
        empty_mask = np.zeros(0, dtype=bool)
        return (input_catalog, empty_mask) if return_removed_mask else input_catalog

    wcs_white = next(iter(image_dict.values()))["wcs"]
    ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(finite):
        removed = np.zeros(len(input_catalog), dtype=bool)
        return (input_catalog, removed) if return_removed_mask else input_catalog

    x = np.full(len(input_catalog), np.nan, dtype=float)
    y = np.full(len(input_catalog), np.nan, dtype=float)
    xw, yw = wcs_white.all_world2pix(ra[finite], dec[finite], 0)
    x[finite] = np.asarray(xw, dtype=float)
    y[finite] = np.asarray(yw, dtype=float)
    finite &= np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        removed = np.zeros(len(input_catalog), dtype=bool)
        return (input_catalog, removed) if return_removed_mask else input_catalog

    iy = np.rint(y).astype(int)
    ix = np.rint(x).astype(int)
    offs = _disk_offsets(radius_pix)

    sat_any = np.zeros(len(input_catalog), dtype=bool)
    sat_all = np.ones(len(input_catalog), dtype=bool)
    has_band = np.zeros(len(input_catalog), dtype=bool)

    for _band, d in image_dict.items():
        sat = d.get("satur_mask", None)
        if sat is None:
            continue
        sat = np.asarray(sat, dtype=bool)
        H, W = sat.shape
        band_hit = np.zeros(len(input_catalog), dtype=bool)
        for dy, dx in offs:
            yy = iy + int(dy)
            xx = ix + int(dx)
            m = finite & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if np.any(m):
                band_hit[m] |= sat[yy[m], xx[m]]
        sat_any |= band_hit
        sat_all &= band_hit
        has_band |= finite

    if require_all_bands:
        sat_hit = has_band & sat_all
        mode = "all-bands"
    else:
        sat_hit = sat_any
        mode = "any-band"

    keep = ~sat_hit
    n_before = len(input_catalog)
    out = input_catalog.loc[keep].copy().reset_index(drop=True)
    n_removed = n_before - len(out)
    _log(f"Saturation-cut ({mode}): removed {n_removed}/{n_before} sources within radius_pix={float(radius_pix):.2f}")
    removed = ~keep
    return (out, removed) if return_removed_mask else out


def _active_epsf_tags_from_catalog(
    *,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, dict],
    epsf_ngrid: int,
) -> set[str]:
    if len(input_catalog) == 0:
        return set()

    col_map = {str(c).strip().lower(): str(c) for c in input_catalog.columns}
    ra_col = col_map.get("ra")
    dec_col = col_map.get("dec")
    if ra_col is None or dec_col is None:
        return set()

    wcs = next(iter(image_dict.values()))["wcs"]
    sample_img = next(iter(image_dict.values()))["img_scaled"]
    ny, nx = sample_img.shape

    ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(finite):
        return set()

    x = np.full(len(input_catalog), np.nan, dtype=float)
    y = np.full(len(input_catalog), np.nan, dtype=float)
    xw, yw = wcs.all_world2pix(ra[finite], dec[finite], 0)
    x[finite] = np.asarray(xw, dtype=float)
    y[finite] = np.asarray(yw, dtype=float)

    inb = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    if not np.any(inb):
        return set()

    x_step = max(1, nx // int(epsf_ngrid))
    y_step = max(1, ny // int(epsf_ngrid))
    ec = np.floor(x[inb] / x_step).astype(int)
    er = np.floor(y[inb] / y_step).astype(int)
    ec = np.clip(ec, 0, int(epsf_ngrid) - 1)
    er = np.clip(er, 0, int(epsf_ngrid) - 1)

    tags = {f"r{int(r):02d}_c{int(c):02d}" for r, c in zip(er, ec)}
    return tags


def load_inputs(cfg: dict) -> dict:
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    checks = cfg["checks"]
    crop_cfg = cfg["crop"]
    sat_cut_cfg = cfg.get("source_saturation_cut", {})
    perf_cfg = cfg.get("performance", {})
    t_load0 = time.perf_counter()
    t_prep = 0.0
    t_white = 0.0
    t_crop = 0.0
    t_sat = 0.0
    t_overlay = 0.0

    def _resolve_workers(raw: object, cap: int, *, label: str) -> int:
        if cap <= 0:
            return 1
        if isinstance(raw, str):
            if raw.strip().lower() == "auto":
                return cap
            try:
                parsed = int(raw)
            except Exception as e:
                raise ValueError(f"{label} must be 'auto' or an integer, got: {raw!r}") from e
            return max(1, min(cap, parsed))
        if isinstance(raw, (int, np.integer)):
            return max(1, min(cap, int(raw)))
        raise TypeError(f"{label} must be 'auto' or an integer, got: {type(raw).__name__}")

    cfg_path = cfg.get("config_path", None)
    if cfg_path is not None:
        _log(f"Config path: {cfg_path}")
    _log(f"Input catalog path: {inputs['input_catalog']}")
    _log(f"GaiaXP synphot path: {inputs.get('gaiaxp_synphot_csv', None)}")

    input_catalog = pd.read_csv(inputs["input_catalog"])
    cols = list(input_catalog.columns)
    col_map = {str(c).strip().lower(): str(c) for c in cols}
    ra_col = col_map.get("ra")
    dec_col = col_map.get("dec")
    type_col = col_map.get("type")
    if ra_col is None or dec_col is None:
        raise ValueError("input_catalog must have RA/DEC columns")
    if ra_col != "RA" or dec_col != "DEC":
        _log(f"[WARN] input_catalog uses {ra_col}/{dec_col} columns; interpreting as Right Ascension / Declination. Expected format is RA/DEC.")
    id_col = col_map.get("id")
    key_cols = [id_col] if id_col is not None else [ra_col, dec_col]
    exclusion_flags = input_catalog[key_cols].copy()
    exclusion_flags["excluded_crop"] = False
    exclusion_flags["excluded_saturation"] = False

    image_list = _read_image_list(inputs["image_list_file"], cfg["config_dir"])
    _log(f"7DT frames to stack: {len(image_list)} (list={inputs['image_list_file']})")
    for image_path in image_list:
        logger.debug("image_path=%s", image_path)
    if not image_list:
        raise FileNotFoundError(f"No image paths found in {inputs['image_list_file']}")

    zp_ref = float(cfg["image_scaling"]["zp_ref"])

    image_dict: dict[str, dict] = {}
    ref_wcs = None
    ref_shape = None
    _log(f"Preparing image arrays/masks/WCS for {len(image_list)} frames...")
    t0 = time.perf_counter()
    prep_workers = _resolve_workers(
        perf_cfg.get("frame_prep_workers", "auto"),
        max(1, min(len(image_list), int(os.cpu_count() or 1))),
        label="performance.frame_prep_workers",
    )
    _log(f"Per-frame preparation workers: {prep_workers}")
    frame_results: dict[Path, dict] = {}

    with cf.ThreadPoolExecutor(max_workers=prep_workers) as ex:
        futs = {ex.submit(_prepare_image_frame, fp, zp_ref): fp for fp in image_list}
        for fut in cf.as_completed(futs):
            fp = futs[fut]
            frame_results[fp] = fut.result()

    for fp in image_list:
        fr = frame_results[fp]
        for msg in fr.get("warn_messages", []):
            _log(msg)

        filt = str(fr["filt"])
        shape = tuple(fr["shape"])
        wcs = fr["wcs"]

        if checks.get("require_same_shape", True):
            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                raise RuntimeError(f"Image shape mismatch: {fp} has {shape}, expected {ref_shape}")

        if checks.get("require_wcs_alignment", True):
            if ref_wcs is None:
                ref_wcs = wcs
            else:
                if not _wcs_close(ref_wcs, wcs, checks.get("wcs_tolerance", {})):
                    _log(f"[CHECK] WCS alignment FAILED for {fp}")
                    raise RuntimeError(f"WCS mismatch with reference: {fp}")

        image_dict[filt] = dict(
            img_scaled=fr["img_scaled"],
            sigma_total_scaled=fr["sigma_total_scaled"],
            sigma_sky_scaled=fr["sigma_sky_scaled"],
            bad=fr["bad"],
            satur_mask=fr["satur_mask"],
            hdr=fr["hdr"],
            wcs=wcs,
            scale=float(fr["scale"]),
            fp=fr["fp"],
        )
    t_prep = time.perf_counter() - t0
    _log(f"Finished per-frame preparation for {len(image_dict)} filters.")

    # Validate that input catalog band columns match image FILTERs, if provided.
    catalog_bands = {
        str(c[len("FLUX_") :]).strip()
        for c in input_catalog.columns
        if isinstance(c, str) and c.startswith("FLUX_") and len(c) > len("FLUX_")
    }
    if catalog_bands:
        image_bands = {str(b).strip() for b in image_dict.keys()}
        missing_in_images = sorted(catalog_bands - image_bands)
        missing_in_catalog = sorted(image_bands - catalog_bands)
        if missing_in_images or missing_in_catalog:
            raise RuntimeError(
                "Band mismatch between input catalog and image list. "
                f"catalog-only={missing_in_images} image-only={missing_in_catalog}"
            )
        _log("[CHECK] Input catalog bands match image FILTERs.")

    _log(f"Filters loaded: {list(image_dict.keys())}")
    if checks.get("require_wcs_alignment", True) and ref_wcs is not None:
        _log("[CHECK] WCS alignment PASSED for all images.")

    _log("Building white stack and white-noise map...")
    t0 = time.perf_counter()
    image_items = [
        (
            np.asarray(d["img_scaled"], dtype=np.float32),
            np.asarray(d["sigma_sky_scaled"], dtype=np.float32),
            np.asarray(d["bad"], dtype=bool),
        )
        for d in image_dict.values()
    ]
    h, w = image_items[0][0].shape
    white_num = np.zeros((h, w), dtype=np.float32)
    white_den = np.zeros((h, w), dtype=np.float32)
    white_mask = np.zeros((h, w), dtype=bool)

    chunk_rows = max(256, min(1024, h // 8 if h >= 8 else h))
    row_ranges = [(y0, min(h, y0 + chunk_rows)) for y0 in range(0, h, chunk_rows)]
    white_workers = _resolve_workers(
        perf_cfg.get("white_stack_workers", "auto"),
        max(1, min(len(row_ranges), int(os.cpu_count() or 1))),
        label="performance.white_stack_workers",
    )
    _log(f"White stack workers: {white_workers} (row chunks={len(row_ranges)})")

    with cf.ThreadPoolExecutor(max_workers=white_workers) as ex:
        futs = [
            ex.submit(_accumulate_white_chunk, image_items, y0, y1)
            for y0, y1 in row_ranges
        ]
        for fut in cf.as_completed(futs):
            y0, y1, num_c, den_c, mask_c = fut.result()
            white_num[y0:y1, :] = num_c
            white_den[y0:y1, :] = den_c
            white_mask[y0:y1, :] = mask_c

    white = np.full_like(white_den, np.nan, dtype=np.float32)
    good = white_den > 0
    white[good] = (white_num[good] / white_den[good]).astype(np.float32)

    sigma_white = np.full_like(white_den, np.inf, dtype=np.float32)
    sigma_white[good] = np.sqrt(1.0 / white_den[good]).astype(np.float32)
    t_white = time.perf_counter() - t0

    _log(f"White made: {white.shape}")

    if crop_cfg.get("enabled", True):
        t0_crop = time.perf_counter()
        CROP_MARGIN = int(crop_cfg.get("margin", 500))
        H0, W0 = white.shape
        x0 = int(CROP_MARGIN)
        x1 = int(W0 - CROP_MARGIN)
        y0 = int(CROP_MARGIN)
        y1 = int(H0 - CROP_MARGIN)

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid crop region: white shape={(H0, W0)} margin={CROP_MARGIN}")

        _log(f"Crop box (pixel, half-open): {(x0, x1, y0, y1)} from white shape={(H0, W0)}")

        crop_out = outputs["cropped_images_dir"]
        crop_out.mkdir(parents=True, exist_ok=True)

        if crop_cfg.get("plot_pre_crop", True):
            _log("Generating pre-crop white diagnostic plot...")
            try:
                DS = int(crop_cfg.get("display_downsample", 4))
                white_show = np.asarray(white)[::DS, ::DS]
                m = np.isfinite(white_show)
                if np.any(m):
                    vmin, vmax = ZScaleInterval().get_limits(white_show[m])
                else:
                    vmin, vmax = None, None

                fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
                ax.imshow(
                    white_show,
                    origin="lower",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    extent=(0, W0 - 1, 0, H0 - 1),
                    rasterized=True,
                )
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2.0))
                ax.set_title(f"White BEFORE crop (DS={DS}) with crop box")
                ax.set_xlim(0, W0 - 1)
                ax.set_ylim(0, H0 - 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.savefig(crop_out / "white_before_crop.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                _log(f"[WARN] Could not plot pre-crop white: {e}")

        wcs0 = next(iter(image_dict.values()))["wcs"]
        _log("Applying crop to catalog and per-band arrays/WCS...")
        ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(ra) & np.isfinite(dec)

        xpix, ypix = wcs0.all_world2pix(ra[ok], dec[ok], 0)
        xpix = np.asarray(xpix, dtype=float)
        ypix = np.asarray(ypix, dtype=float)

        in_crop = (xpix >= x0) & (xpix < x1) & (ypix >= y0) & (ypix < y1)

        keep = np.zeros(len(input_catalog), dtype=bool)
        keep[np.where(ok)[0][in_crop]] = True

        n0 = len(input_catalog)
        removed_crop_keys = input_catalog.loc[~keep, key_cols].copy()
        if len(removed_crop_keys) > 0:
            tmp = removed_crop_keys.copy()
            tmp["__excluded_crop"] = True
            tmp = tmp.drop_duplicates(subset=key_cols)
            exclusion_flags = exclusion_flags.merge(tmp, on=key_cols, how="left")
            exclusion_flags["excluded_crop"] = exclusion_flags["excluded_crop"] | exclusion_flags["__excluded_crop"].fillna(False)
            exclusion_flags = exclusion_flags.drop(columns=["__excluded_crop"])
        input_catalog = input_catalog.loc[keep].copy().reset_index(drop=True)
        _log(f"input_catalog: {n0} -> {len(input_catalog)} rows after crop filtering")

        white = np.asarray(white)[y0:y1, x0:x1]
        sigma_white = np.asarray(sigma_white)[y0:y1, x0:x1]
        if white_mask is not None:
            try:
                white_mask = np.asarray(white_mask)[y0:y1, x0:x1]
            except Exception:
                pass

        for band, d in image_dict.items():
            d["img_scaled"] = np.asarray(d["img_scaled"])[y0:y1, x0:x1]
            d["sigma_total_scaled"] = np.asarray(d["sigma_total_scaled"])[y0:y1, x0:x1]
            d["sigma_sky_scaled"] = np.asarray(d["sigma_sky_scaled"])[y0:y1, x0:x1]
            d["bad"] = np.asarray(d["bad"])[y0:y1, x0:x1]
            if d.get("satur_mask", None) is not None:
                d["satur_mask"] = np.asarray(d["satur_mask"])[y0:y1, x0:x1]

            try:
                new_wcs = d["wcs"].slice((slice(y0, y1), slice(x0, x1)))
                d["wcs"] = new_wcs

                if "hdr" in d and d["hdr"] is not None:
                    new_hdr = d["hdr"].copy()
                    new_hdr.update(new_wcs.to_header(relax=True))
                    new_hdr["NAXIS1"] = int(x1 - x0)
                    new_hdr["NAXIS2"] = int(y1 - y0)
                    d["hdr"] = new_hdr
            except Exception as e:
                _log(f"[WARN] Could not slice WCS / update header for band={band}: {e}")

        _log(f"After crop: white shape={white.shape}")
        _log(f"After crop: first band shape={next(iter(image_dict.values()))['img_scaled'].shape}")

        try:
            _log("Generating post-crop white diagnostic plot...")
            ds_post = max(1, int(crop_cfg.get("post_crop_display_downsample", 1)))
            white_show = np.asarray(white)[::ds_post, ::ds_post]
            m = np.isfinite(white_show)
            if np.any(m):
                vmin, vmax = ZScaleInterval().get_limits(white_show[m])
            else:
                vmin, vmax = None, None
            fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
            ax.imshow(
                white_show,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                extent=(0, white.shape[1] - 1, 0, white.shape[0] - 1),
                rasterized=True,
            )
            ax.set_title(f"White AFTER crop (DS={ds_post})")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(crop_out / "white_after_crop.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            _log(f"[WARN] Could not plot post-crop white: {e}")
        t_crop = time.perf_counter() - t0_crop

    if bool(sat_cut_cfg.get("enabled", False)):
        t0_sat = time.perf_counter()
        _log("Applying saturation-cut filtering on input catalog...")
        filtered_catalog, sat_removed_mask = _filter_sources_near_saturation(
            input_catalog=input_catalog,
            image_dict=image_dict,
            ra_col=ra_col,
            dec_col=dec_col,
            radius_pix=float(sat_cut_cfg.get("radius_pix", 2.0)),
            require_all_bands=bool(sat_cut_cfg.get("require_all_bands", False)),
            return_removed_mask=True,
        )
        if len(filtered_catalog) != len(input_catalog):
            removed_sat_keys = input_catalog.loc[sat_removed_mask, key_cols].copy()
            if len(removed_sat_keys) > 0:
                tmp = removed_sat_keys.copy()
                tmp["__excluded_saturation"] = True
                tmp = tmp.drop_duplicates(subset=key_cols)
                exclusion_flags = exclusion_flags.merge(tmp, on=key_cols, how="left")
                exclusion_flags["excluded_saturation"] = (
                    exclusion_flags["excluded_saturation"] | exclusion_flags["__excluded_saturation"].fillna(False)
                )
                exclusion_flags = exclusion_flags.drop(columns=["__excluded_saturation"])
        input_catalog = filtered_catalog
        t_sat = time.perf_counter() - t0_sat

    if crop_cfg.get("overlay_catalog", True):
        t0_overlay = time.perf_counter()
        _log("Generating white+catalog overlay plot...")
        wcs_white = next(iter(image_dict.values()))["wcs"]
        H, W = white.shape
        fallback_model = str(cfg.get("patch_run", {}).get("gal_model", "dev")).upper()

        ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
        if type_col is not None:
            type_series = input_catalog[type_col].astype(str).str.strip().str.upper()
            type_series = type_series.replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN", "NULL": "UNKNOWN"})
            type_str = type_series.to_numpy()
        else:
            type_str = np.full(len(input_catalog), "UNKNOWN", dtype=object)

        ok = np.isfinite(ra) & np.isfinite(dec)

        xpix, ypix = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
        xpix = np.asarray(xpix, dtype=float)
        ypix = np.asarray(ypix, dtype=float)

        inb = (xpix >= 0) & (xpix < W) & (ypix >= 0) & (ypix < H)

        x_ok = xpix[inb]
        y_ok = ypix[inb]
        type_ok = type_str[ok][inb]

        is_star = type_ok == "STAR"
        is_gal = np.isin(type_ok, ["GAL", "EXP", "DEV", "SERSIC"])
        is_unknown = ~(is_star | is_gal)

        DS_FULL = int(crop_cfg.get("overlay_downsample_full", 1))

        def _plot_white_with_overlay(*, white_arr: np.ndarray, extent: tuple[float, float, float, float], xlim, ylim, title: str):
            interval = ZScaleInterval()
            m = np.isfinite(white_arr)
            if np.any(m):
                vmin, vmax = interval.get_limits(white_arr[m])
            else:
                vmin, vmax = None, None

            fig = plt.figure(figsize=(10, 7), constrained_layout=True)
            ax = fig.add_subplot(111, projection=wcs_white)
            ax.imshow(
                white_arr,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                extent=extent,
                rasterized=True,
            )

            ax.coords[0].set_axislabel("RA", fontsize=12)
            ax.coords[1].set_axislabel("Dec", fontsize=12)

            secax_x = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
            secax_y = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
            secax_x.set_xlabel("x [pix]", fontsize=12)
            secax_y.set_ylabel("y [pix]", fontsize=12)

            if np.any(is_star):
                ax.scatter(
                    x_ok[is_star],
                    y_ok[is_star],
                    s=20,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=0.8,
                    label=f"STAR (N={is_star.sum()})",
                    transform=ax.get_transform("pixel"),
                )
            if np.any(is_gal):
                ax.scatter(
                    x_ok[is_gal],
                    y_ok[is_gal],
                    s=20,
                    facecolors="none",
                    edgecolors="magenta",
                    linewidths=0.8,
                    label=f"GAL (N={is_gal.sum()})",
                    transform=ax.get_transform("pixel"),
                )
            if np.any(is_unknown):
                ax.scatter(
                    x_ok[is_unknown],
                    y_ok[is_unknown],
                    s=24,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=0.9,
                    marker="s",
                    label=f"UNKNOWN (N={is_unknown.sum()})\n(Fallback to {fallback_model})",
                    transform=ax.get_transform("pixel"),
                )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.legend(loc="upper right", frameon=True, fontsize=12)
            return fig

        white_full = np.asarray(white)[::DS_FULL, ::DS_FULL]
        fig = _plot_white_with_overlay(
            white_arr=white_full,
            extent=(0, W - 1, 0, H - 1),
            xlim=(0, W - 1),
            ylim=(0, H - 1),
            title=None,
        )
        try:
            outpath = outputs["cropped_images_dir"] / "white_overlay.png"
            fig.savefig(outpath, dpi=150)
        finally:
            plt.close(fig)
        t_overlay = time.perf_counter() - t0_overlay

    wcs_fits = None
    if image_dict:
        wcs_fits = next(iter(image_dict.values())).get("fp", None)

    state = dict(
        input_catalog=input_catalog,
        image_dict=image_dict,
        white=white,
        sigma_white=sigma_white,
        wcs_fits=wcs_fits,
        exclusion_flags=exclusion_flags,
    )
    t_total = time.perf_counter() - t_load0
    _log(
        "load_inputs timing [s]: prep=%.2f white=%.2f crop=%.2f sat=%.2f overlay=%.2f total=%.2f"
        % (t_prep, t_white, t_crop, t_sat, t_overlay, t_total)
    )
    return state


def build_epsf_from_config(cfg: dict, state: dict) -> Path:
    epsf_cfg = cfg["epsf"]
    outputs = cfg["outputs"]

    epsf_conf = EPSFConfig(
        epsf_ngrid=int(epsf_cfg["epsf_ngrid"]),
        ngrid=int(epsf_cfg["ngrid"]),
        thresh_sigma=float(epsf_cfg["thresh_sigma"]),
        minarea=int(epsf_cfg["minarea"]),
        cutout_size=int(epsf_cfg["cutout_size"]),
        edge_pad=int(epsf_cfg["edge_pad"]),
        min_sep=float(epsf_cfg["min_sep"]),
        max_stars=int(epsf_cfg["max_stars"]),
        q_range=(float(epsf_cfg["q_range"][0]), float(epsf_cfg["q_range"][1])),
        psfstar_mode=str(epsf_cfg["psfstar_mode"]),
        gaia_mag_min=float(epsf_cfg["gaia_mag_min"]),
        gaia_mag_max=float(epsf_cfg["gaia_mag_max"]),
        gaia_snap_to_sep=bool(epsf_cfg["gaia_snap_to_sep"]),
        gaia_forced_k_fwhm=float(epsf_cfg["gaia_forced_k_fwhm"]),
        gaia_forced_rmin_pix=float(epsf_cfg["gaia_forced_rmin_pix"]),
        gaia_forced_rmax_pix=float(epsf_cfg["gaia_forced_rmax_pix"]),
        gaia_snr_min=float(epsf_cfg["gaia_snr_min"]),
        gaia_match_r_pix=float(epsf_cfg["gaia_match_r_pix"]),
        gaia_seed_sat_r=int(epsf_cfg["gaia_seed_sat_r"]),
        gaia_seed_sat_div=float(epsf_cfg["gaia_seed_sat_div"]),
        oversamp=int(epsf_cfg["oversamp"]),
        maxiters=int(epsf_cfg["maxiters"]),
        do_clip=bool(epsf_cfg["do_clip"]),
        recenter_boxsize=int(epsf_cfg["recenter_boxsize"]),
        final_psf_size=int(epsf_cfg["final_psf_size"]),
        save_star_montage_max=int(epsf_cfg["save_star_montage_max"]),
    )

    gaiaxp = None
    if bool(epsf_cfg.get("use_gaiaxp", True)):
        gaia_path = cfg["inputs"]["gaiaxp_synphot_csv"]
        if gaia_path and Path(gaia_path).exists():
            gaiaxp = pd.read_csv(gaia_path)
            _log(f"GaiaXP loaded: {len(gaiaxp)} rows from {gaia_path}")
        else:
            _log(f"[WARN] GaiaXP catalog not found: {gaia_path}")

    max_workers = epsf_cfg.get("max_workers", "auto")
    max_workers_val = None if str(max_workers).lower() == "auto" else int(max_workers)
    active_epsf_tags = None
    if bool(epsf_cfg.get("skip_empty_epsf_patches", True)):
        active_epsf_tags = _active_epsf_tags_from_catalog(
            input_catalog=state["input_catalog"],
            image_dict=state["image_dict"],
            epsf_ngrid=int(epsf_cfg["epsf_ngrid"]),
        )
        state["active_epsf_tags"] = active_epsf_tags
        _log(
            f"Active ePSF cells from input catalog: {len(active_epsf_tags)}/{int(epsf_cfg['epsf_ngrid']) ** 2}"
        )
        if len(active_epsf_tags) == 0:
            _log("[WARN] No active ePSF cells found from input catalog; ePSF generation will be skipped.")

    return _build_epsf_impl(
        cfg=epsf_conf,
        outputs_dir=outputs["epsf_dir"],
        image_dict=state["image_dict"],
        gaiaxp=gaiaxp,
        use_gaiaxp=bool(epsf_cfg.get("use_gaiaxp", True)),
        parallel_bands=bool(epsf_cfg.get("parallel_bands", True)),
        max_workers=max_workers_val,
        active_epsf_tags=active_epsf_tags,
    )


def build_patches_from_config(cfg: dict, state: dict) -> Path:
    epsf_cfg = cfg["epsf"]
    patches_cfg = cfg["patches"]
    outputs = cfg["outputs"]

    image_shape = next(iter(state["image_dict"].values()))["img_scaled"].shape
    active_epsf_tags = None
    if bool(epsf_cfg.get("skip_empty_epsf_patches", True)):
        active_epsf_tags = state.get("active_epsf_tags")
        if active_epsf_tags is None:
            active_epsf_tags = _active_epsf_tags_from_catalog(
                input_catalog=state["input_catalog"],
                image_dict=state["image_dict"],
                epsf_ngrid=int(epsf_cfg["epsf_ngrid"]),
            )
        _log(
            f"Building patches from active ePSF cells: {len(active_epsf_tags)}/{int(epsf_cfg['epsf_ngrid']) ** 2}"
        )

    return _build_patches_impl(
        epsf_ngrid=int(epsf_cfg["epsf_ngrid"]),
        patch_ngrid=int(epsf_cfg["ngrid"]),
        final_psf_size=int(epsf_cfg["final_psf_size"]),
        halo_pix_min=int(patches_cfg["halo_pix_min"]),
        image_shape=image_shape,
        out_dir=outputs["patches_dir"],
        active_epsf_tags=active_epsf_tags,
    )


def build_patch_inputs_from_config(cfg: dict, state: dict) -> list[dict]:
    patch_inputs_cfg = cfg["patch_inputs"]
    outputs = cfg["outputs"]

    patch_def_path = outputs["patches_dir"] / "patches.json"

    return _build_patch_inputs_impl(
        patch_def_path=patch_def_path,
        out_dir=outputs["patch_inputs_dir"],
        image_dict=state["image_dict"],
        input_catalog=state["input_catalog"],
        include_wcs_in_payload=bool(patch_inputs_cfg["include_wcs_in_payload"]),
        save_patch_inputs=bool(patch_inputs_cfg["save_patch_inputs"]),
        skip_empty_patch=bool(patch_inputs_cfg["skip_empty_patch"]),
        max_workers=int(patch_inputs_cfg["max_workers"]),
        gzip_compresslevel=int(patch_inputs_cfg["gzip_compresslevel"]),
        keep_payloads_in_memory=bool(patch_inputs_cfg["keep_payloads_in_memory"]),
    )


def run_patch_subprocesses(cfg: dict) -> Path:
    patch_run_cfg = cfg["patch_run"]
    outputs = cfg["outputs"]
    moffat_cfg = cfg.get("moffat_psf", {})

    extra_args = [
        "--cutout-size",
        str(int(patch_run_cfg["cutout_size"])),
        "--cutout-start",
        str(int(patch_run_cfg["cutout_start"])),
        "--cutout-data-p-lo",
        str(float(patch_run_cfg["cutout_data_p_lo"])),
        "--cutout-data-p-hi",
        str(float(patch_run_cfg["cutout_data_p_hi"])),
        "--cutout-model-p-lo",
        str(float(patch_run_cfg["cutout_model_p_lo"])),
        "--cutout-model-p-hi",
        str(float(patch_run_cfg["cutout_model_p_hi"])),
        "--cutout-resid-p-abs",
        str(float(patch_run_cfg["cutout_resid_p_abs"])),
        "--gal-model",
        str(patch_run_cfg["gal_model"]),
        "--n-opt-iters",
        str(int(patch_run_cfg["n_opt_iters"])),
        "--dlnp-stop",
        str(float(patch_run_cfg["dlnp_stop"])),
        "--flag-maxiter-margin",
        str(int(patch_run_cfg["flag_maxiter_margin"])),
        "--r-ap",
        str(float(patch_run_cfg["r_ap"])),
        "--eps-flux",
        str(float(patch_run_cfg["eps_flux"])),
        "--psf-model",
        str(patch_run_cfg.get("psf_model", "hybrid")),
        "--psf-fallback-model",
        str(patch_run_cfg.get("psf_fallback_model", "gaussian_mixture")),
        "--min-epsf-nstars-for-use",
        str(int(patch_run_cfg.get("min_epsf_nstars_for_use", 5))),
        "--ncircular-gaussian-n",
        str(int(patch_run_cfg.get("ncircular_gaussian_n", 3))),
        "--hybrid-psf-n",
        str(int(patch_run_cfg.get("hybrid_psf_n", 4))),
        "--gaussian-mixture-n",
        str(int(patch_run_cfg.get("gaussian_mixture_n", 4))),
        "--sersic-n-init",
        str(float(patch_run_cfg["sersic_n_init"])),
        "--re-fallback-pix",
        str(float(patch_run_cfg["re_fallback_pix"])),
        "--fallback-default-fwhm-pix",
        str(float(patch_run_cfg.get("fallback_default_fwhm_pix", 4.0))),
        "--fallback-stamp-radius-pix",
        str(float(patch_run_cfg.get("fallback_stamp_radius_pix", 25.0))),
        "--moffat-beta",
        str(float(moffat_cfg.get("beta", 3.5))),
        "--moffat-radius-pix",
        str(float(moffat_cfg.get("radius_pix", 25.0))),
    ]
    if patch_run_cfg.get("cutout_max_sources") is not None:
        extra_args.extend(["--cutout-max-sources", str(int(patch_run_cfg["cutout_max_sources"]))])
    if bool(patch_run_cfg.get("no_cutouts", False)):
        extra_args.append("--no-cutouts")
    if bool(patch_run_cfg.get("no_patch_overview", False)):
        extra_args.append("--no-patch-overview")

    return run_subprocesses(
        patch_input_dir=outputs["patch_inputs_dir"],
        epsf_root=outputs["epsf_dir"],
        outdir=outputs["tractor_out_dir"],
        python_exe=str(patch_run_cfg["python_exe"]),
        max_workers=int(patch_run_cfg["max_workers"]),
        threads_per_process=int(patch_run_cfg["threads_per_process"]),
        resume=bool(patch_run_cfg.get("resume", False)),
        extra_args=extra_args,
    )


def merge_results(cfg: dict, state: dict | None = None) -> Path:
    outputs = cfg["outputs"]
    merge_cfg = cfg["merge"]

    wcs_fits = merge_cfg.get("wcs_fits")
    if wcs_fits is None and state is not None:
        wcs_fits = state.get("wcs_fits")
    exclusion_flags = state.get("exclusion_flags") if state is not None else None
    return merge_catalogs(
        input_catalog=Path(cfg["inputs"]["input_catalog"]),
        patch_outdir=outputs["tractor_out_dir"],
        out_path=outputs["final_catalog"],
        pattern=str(merge_cfg["pattern"]),
        wcs_fits=Path(wcs_fits) if wcs_fits else None,
        exclusion_flags=exclusion_flags,
    )


def build_epsf(cfg: dict, state: dict) -> Path:
    return build_epsf_from_config(cfg, state)


def build_patches(cfg: dict, state: dict) -> Path:
    return build_patches_from_config(cfg, state)


def build_patch_inputs(cfg: dict, state: dict) -> list[dict]:
    return build_patch_inputs_from_config(cfg, state)


def run_pipeline(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    ignore_warnings = bool(log_cfg.get("ignore_warnings", True))
    if ignore_warnings:
        warnings.filterwarnings("ignore")
        _log("Warnings are suppressed (logging.ignore_warnings=true).")
    else:
        warnings.filterwarnings("default")
        _log("Warnings are enabled.")

    t_total = time.perf_counter()

    def _run_step(name: str, func):
        _log(f"Starting {name}...")
        t0 = time.perf_counter()
        result = func()
        _log(f"Finished {name} in {_format_elapsed(time.perf_counter() - t0)}")
        return result

    state = _run_step("load inputs + validation", lambda: load_inputs(cfg))
    _run_step("build ePSF", lambda: build_epsf(cfg, state))
    _run_step("build patches", lambda: build_patches(cfg, state))
    _run_step("build patch inputs", lambda: build_patch_inputs(cfg, state))
    _run_step("run patch subprocesses", lambda: run_patch_subprocesses(cfg))
    final_catalog = _run_step("merge patch catalogs", lambda: merge_results(cfg, state))
    _summarize_fit_results(Path(final_catalog))
    _log(f"Pipeline finished in {_format_elapsed(time.perf_counter() - t_total)}")
