from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.patch_inputs")


def build_patch_inputs(
    *,
    patch_def_path: Path,
    out_dir: Path,
    image_dict: dict,
    input_catalog: pd.DataFrame,
    include_wcs_in_payload: bool,
    save_patch_inputs: bool,
    skip_empty_patch: bool,
    max_workers: int,
    gzip_compresslevel: int,
    keep_payloads_in_memory: bool,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert patch_def_path.exists(), f"Missing patch defs: {patch_def_path}"

    patch_defs = json.loads(patch_def_path.read_text())
    logger.info("Loaded patch defs: %d", len(patch_defs))
    logger.info("SKIP_EMPTY_PATCH: %s", skip_empty_patch)
    logger.info(
        "SAVE_PATCH_INPUTS: %s | MAX_WORKERS_PATCH_INPUTS: %d | GZIP_COMPRESSLEVEL: %d",
        save_patch_inputs,
        int(max_workers),
        int(gzip_compresslevel),
    )

    N_TOTAL = len(patch_defs)
    N_DONE = 0
    N_KEEP = 0
    N_SKIP = 0
    PROGRESS_EVERY = max(1, N_TOTAL // 20)
    use_inline_progress = sys.stderr.isatty()
    tstart = time.time()
    last_non_tty_log = 0.0
    inline_width = 160

    def _progress_bar(done: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return "-" * width
        frac = max(0.0, min(1.0, float(done) / float(total)))
        filled = int(round(frac * width))
        return "#" * filled + "-" * (width - filled)

    def _emit_processing_progress(*, force_log: bool = False, queued_writes: int = 0) -> None:
        nonlocal last_non_tty_log
        msg = (
            f"Patch inputs build [{_progress_bar(N_DONE, N_TOTAL)}] "
            f"{N_DONE}/{N_TOTAL} keep={N_KEEP} skip={N_SKIP} queued={queued_writes} "
            f"(total_patches={N_TOTAL})"
        )
        if use_inline_progress:
            sys.stderr.write("\r" + msg.ljust(inline_width))
            sys.stderr.flush()
        else:
            now = time.time()
            if force_log or (now - last_non_tty_log) >= 10.0:
                logger.info(msg)
                last_non_tty_log = now

    def _emit_write_progress(*, done_write: int, total_write: int, force_log: bool = False) -> None:
        nonlocal last_non_tty_log
        elapsed = max(1e-6, time.time() - tstart)
        rate = done_write / elapsed
        msg = (
            f"Patch inputs write [{_progress_bar(done_write, total_write)}] "
            f"{done_write}/{total_write} rate={rate:.2f}/s "
            f"(kept={total_write} total={N_TOTAL})"
        )
        if use_inline_progress:
            sys.stderr.write("\r" + msg.ljust(inline_width))
            sys.stderr.flush()
        else:
            now = time.time()
            if force_log or (now - last_non_tty_log) >= 10.0:
                logger.info(msg)
                last_non_tty_log = now

    cols = list(input_catalog.columns)
    col_map = {str(c).strip().lower(): str(c) for c in cols}
    _ra_col = col_map.get("ra")
    _dec_col = col_map.get("dec")
    if _ra_col is None or _dec_col is None:
        raise ValueError("input_catalog must have RA/DEC columns")
    if _ra_col != "RA" or _dec_col != "DEC":
        logger.warning(
            "input_catalog uses %s/%s columns; interpreting as Right Ascension / Declination. Expected format is RA/DEC.",
            _ra_col,
            _dec_col,
        )

    ra = pd.to_numeric(input_catalog[_ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(input_catalog[_dec_col], errors="coerce").to_numpy(dtype=float)

    wcs_white = next(iter(image_dict.values()))["wcs"]

    ok = np.isfinite(ra) & np.isfinite(dec)
    xpix = np.full(len(ra), np.nan, dtype=float)
    ypix = np.full(len(dec), np.nan, dtype=float)
    if np.any(ok):
        xw, yw = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
        xpix[ok] = xw
        ypix[ok] = yw

    cat_global = input_catalog.copy()
    cat_global["x_pix_white"] = xpix
    cat_global["y_pix_white"] = ypix

    patch_inputs = []
    patch_payloads = [] if keep_payloads_in_memory else None

    xg_all = cat_global["x_pix_white"].to_numpy(dtype=float)
    yg_all = cat_global["y_pix_white"].to_numpy(dtype=float)

    _band_items = [(str(band), d) for band, d in image_dict.items()]

    def _build_payload_for_patch(pdef: dict):
        x0b, x1b, y0b, y1b = pdef["x0_base"], pdef["x1_base"], pdef["y0_base"], pdef["y1_base"]
        x0r, x1r, y0r, y1r = pdef["x0_roi"], pdef["x1_roi"], pdef["y0_roi"], pdef["y1_roi"]

        in_base = (
            np.isfinite(xg_all)
            & np.isfinite(yg_all)
            & (xg_all >= x0b)
            & (xg_all < x1b)
            & (yg_all >= y0b)
            & (yg_all < y1b)
        )
        cat_patch = cat_global.loc[in_base].copy().reset_index(drop=True)
        if skip_empty_patch and len(cat_patch) == 0:
            return None

        cat_patch["x_pix_patch"] = cat_patch["x_pix_white"] - float(x0r)
        cat_patch["y_pix_patch"] = cat_patch["y_pix_white"] - float(y0r)

        per_filter_patch = {}
        for band, d in _band_items:
            img_full = d["img_scaled"]
            sig_total = d["sigma_total_scaled"]
            sig_sky = d["sigma_sky_scaled"]
            bad_full = d.get("bad", None)
            if bad_full is None:
                bad_full = ~np.isfinite(img_full)

            entry = dict(
                img_scaled=img_full[y0r:y1r, x0r:x1r],
                sigma_total_scaled=sig_total[y0r:y1r, x0r:x1r],
                sigma_sky_scaled=sig_sky[y0r:y1r, x0r:x1r],
                bad=np.asarray(bad_full[y0r:y1r, x0r:x1r], dtype=bool),
                scale=float(d.get("scale", 1.0)),
                fp=str(d.get("fp", "")),
                x0w=int(x0r),
                x1w=int(x1r),
                y0w=int(y0r),
                y1w=int(y1r),
            )

            if include_wcs_in_payload:
                wcs_patch = d["wcs"].slice((slice(y0r, y1r), slice(x0r, x1r)))
                entry["wcs_patch"] = wcs_patch
                entry["wcs_full"] = d["wcs"]

            per_filter_patch[band] = entry

        meta = dict(
            patch_tag=pdef["patch_tag"],
            epsf_tag=pdef["epsf_tag"],
            epsf_r=int(pdef["epsf_r"]),
            epsf_c=int(pdef["epsf_c"]),
            patch_r=int(pdef["patch_r"]),
            patch_c=int(pdef["patch_c"]),
            halo_pix=int(pdef["halo_pix"]),
            base_bbox=(int(x0b), int(x1b), int(y0b), int(y1b)),
            roi_bbox=(int(x0r), int(x1r), int(y0r), int(y1r)),
            n_sources=int(len(cat_patch)),
        )

        payload = dict(per_filter_patch=per_filter_patch, cat_patch=cat_patch, meta=meta)
        return meta, payload

    def _write_payload(meta: dict, payload: dict):
        outpath = out_dir / f"{meta['patch_tag']}.pkl.gz"
        with gzip.open(outpath, "wb", compresslevel=int(gzip_compresslevel)) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return str(outpath)

    if save_patch_inputs and int(max_workers) > 1:
        futs = {}
        with cf.ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            _emit_processing_progress(force_log=True, queued_writes=0)
            for pdef in patch_defs:
                N_DONE += 1
                built = _build_payload_for_patch(pdef)
                if built is None:
                    N_SKIP += 1
                    if (N_DONE % PROGRESS_EVERY) == 0 or N_DONE == N_TOTAL:
                        _emit_processing_progress(force_log=True, queued_writes=len(futs))
                    continue
                meta, payload = built
                patch_inputs.append(meta)
                if keep_payloads_in_memory:
                    patch_payloads.append(payload)
                N_KEEP += 1
                futs[ex.submit(_write_payload, meta, payload)] = meta["patch_tag"]

                if (N_DONE % PROGRESS_EVERY) == 0 or N_DONE == N_TOTAL:
                    _emit_processing_progress(force_log=True, queued_writes=len(futs))

            ndone_w = 0
            pending = set(futs.keys())
            total_writes = len(pending)
            _emit_write_progress(done_write=ndone_w, total_write=total_writes, force_log=True)
            while pending:
                done_now, pending = cf.wait(pending, timeout=0.5, return_when=cf.FIRST_COMPLETED)
                if not done_now:
                    _emit_write_progress(done_write=ndone_w, total_write=total_writes, force_log=False)
                    continue
                for fut in done_now:
                    _ = fut.result()
                    ndone_w += 1
                _emit_write_progress(done_write=ndone_w, total_write=total_writes, force_log=True)
    else:
        _emit_processing_progress(force_log=True, queued_writes=0)
        for pdef in patch_defs:
            N_DONE += 1
            built = _build_payload_for_patch(pdef)
            if built is None:
                N_SKIP += 1
                if (N_DONE % PROGRESS_EVERY) == 0 or N_DONE == N_TOTAL:
                    _emit_processing_progress(force_log=True, queued_writes=0)
                continue

            meta, payload = built
            patch_inputs.append(meta)
            if keep_payloads_in_memory:
                patch_payloads.append(payload)
            N_KEEP += 1

            if save_patch_inputs:
                _ = _write_payload(meta, payload)

            if (N_DONE % PROGRESS_EVERY) == 0 or N_DONE == N_TOTAL:
                _emit_processing_progress(force_log=True, queued_writes=0)

    if use_inline_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    logger.info("Patch inputs created: %d", len(patch_inputs))
    if save_patch_inputs:
        logger.info("Saved to: %s", out_dir)
    else:
        logger.info("Saved to: (skipped)")

    return patch_inputs
