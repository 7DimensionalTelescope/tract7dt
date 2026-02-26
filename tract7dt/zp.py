"""
Zero-point calibration using GaiaXP synthetic photometry.

Two entry points:
  - augment_catalog_with_gaia(): inject Gaia synphot sources into the input catalog
  - compute_zp(): derive per-band ZP from fitted Gaia stars, apply AB magnitudes
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger("tract7dt.zp")

K_MAG = 2.5 / np.log(10.0)


# ---------------------------------------------------------------------------
#  Gaia catalog augmentation
# ---------------------------------------------------------------------------

def _compute_gaia_selection_box(
    *,
    x_sources: np.ndarray,
    y_sources: np.ndarray,
    min_box_size: int,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Compute a square bounding box that contains all source positions.

    The box is expanded to at least min_box_size x min_box_size (centered on the
    source centroid), shifted to keep square shape at image edges, and clamped to
    image bounds only when the image itself is smaller.

    Returns (x0, x1, y0, y1) in pixel coordinates.
    """
    ok = np.isfinite(x_sources) & np.isfinite(y_sources)
    if not np.any(ok):
        cx, cy = img_w / 2.0, img_h / 2.0
    else:
        xmin = float(np.nanmin(x_sources[ok]))
        xmax = float(np.nanmax(x_sources[ok]))
        ymin = float(np.nanmin(y_sources[ok]))
        ymax = float(np.nanmax(y_sources[ok]))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        box_w = xmax - xmin
        box_h = ymax - ymin
        min_box_size = max(min_box_size, int(np.ceil(max(box_w, box_h))))

    half = min_box_size / 2.0
    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > img_w:
        x0 -= (x1 - img_w)
        x1 = img_w
    if y1 > img_h:
        y0 -= (y1 - img_h)
        y1 = img_h

    x0 = max(0, int(np.floor(x0)))
    x1 = min(img_w, int(np.ceil(x1)))
    y0 = max(0, int(np.floor(y0)))
    y1 = min(img_h, int(np.ceil(y1)))

    return x0, x1, y0, y1


def _save_augmentation_overlay(
    *,
    white: np.ndarray,
    wcs: Any,
    x_orig: np.ndarray,
    y_orig: np.ndarray,
    x_gaia_new: np.ndarray,
    y_gaia_new: np.ndarray,
    x_gaia_matched: np.ndarray,
    y_gaia_matched: np.ndarray,
    x_gaia_sat: np.ndarray,
    y_gaia_sat: np.ndarray,
    excluded_crop: np.ndarray | None = None,
    excluded_saturation: np.ndarray | None = None,
    bbox: tuple[int, int, int, int],
    outpath: Path,
    dpi: int = 150,
) -> None:
    """Overlay original + injected Gaia sources and the selection box on the white image."""
    from astropy.visualization import ZScaleInterval

    outpath.parent.mkdir(parents=True, exist_ok=True)

    H, W = white.shape
    interval = ZScaleInterval()
    finite = white[np.isfinite(white)]
    vmin, vmax = interval.get_limits(finite) if len(finite) > 0 else (0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
    ax.imshow(white, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    ok_o = np.isfinite(x_orig) & np.isfinite(y_orig)
    if np.any(ok_o):
        ax.scatter(x_orig[ok_o], y_orig[ok_o], s=20, marker="o", facecolors="none",
                   edgecolors="cyan", linewidths=0.8, alpha=0.8, label=f"original ({int(ok_o.sum())})")

    ok_m = np.isfinite(x_gaia_matched) & np.isfinite(y_gaia_matched)
    if np.any(ok_m):
        ax.scatter(x_gaia_matched[ok_m], y_gaia_matched[ok_m], s=30, marker="s", facecolors="none",
                   edgecolors="lime", linewidths=0.8, alpha=0.8, label=f"Gaia matched ({int(ok_m.sum())})")

    ok_n = np.isfinite(x_gaia_new) & np.isfinite(y_gaia_new)
    if np.any(ok_n):
        ax.scatter(x_gaia_new[ok_n], y_gaia_new[ok_n], s=20, marker="o", facecolors="none",
                   edgecolors="magenta", linewidths=0.8, alpha=0.8, label=f"Gaia injected ({int(ok_n.sum())})")

    ok_s = np.isfinite(x_gaia_sat) & np.isfinite(y_gaia_sat)
    if np.any(ok_s):
        ax.scatter(x_gaia_sat[ok_s], y_gaia_sat[ok_s], s=40, marker="x",
                   c="red", linewidths=1.0, alpha=0.9, label=f"Gaia saturated ({int(ok_s.sum())})")

    if excluded_saturation is not None:
        sat_mask = excluded_saturation & ok_o
        n_sat = int(sat_mask.sum())
        if n_sat > 0:
            ax.scatter(x_orig[sat_mask], y_orig[sat_mask], s=60, marker="x",
                       c="red", linewidths=1.2, alpha=0.9, zorder=5,
                       label=f"excluded: saturation ({n_sat})")

    if excluded_crop is not None:
        crop_mask = excluded_crop & ok_o
        n_crop = int(crop_mask.sum())
        if n_crop > 0:
            ax.scatter(x_orig[crop_mask], y_orig[crop_mask], s=60, marker="x",
                       c="orange", linewidths=1.2, alpha=0.9, zorder=5,
                       label=f"excluded: crop ({n_crop})")

    x0, x1, y0, y1 = bbox
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5,
                          edgecolor="yellow", facecolor="none", linestyle="--", label="selection box")
    ax.add_patch(rect)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_title("Gaia augmentation overlay on white stack")
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote Gaia augmentation overlay: %s", outpath)


def _filter_gaia_saturation(
    gaia_df: pd.DataFrame,
    *,
    image_dict: dict,
    wcs: Any,
    radius_pix: float,
    require_all_bands: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Remove Gaia sources near saturated pixels, mirroring the pipeline's saturation cut."""
    ra = gaia_df["ra"].to_numpy(dtype=float)
    dec = gaia_df["dec"].to_numpy(dtype=float)
    x, y = wcs.all_world2pix(ra, dec, 0)
    ix = np.rint(x).astype(int)
    iy = np.rint(y).astype(int)

    r = max(1, int(np.ceil(radius_pix)))
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
               if dy * dy + dx * dx <= radius_pix * radius_pix]

    sat_any = np.zeros(len(gaia_df), dtype=bool)
    sat_all = np.ones(len(gaia_df), dtype=bool)
    has_band = np.zeros(len(gaia_df), dtype=bool)
    finite = np.isfinite(x) & np.isfinite(y)

    for _band, d in image_dict.items():
        sat = d.get("satur_mask", None)
        if sat is None:
            continue
        sat = np.asarray(sat, dtype=bool)
        H, W = sat.shape
        band_hit = np.zeros(len(gaia_df), dtype=bool)
        for dy, dx in offsets:
            yy = iy + int(dy)
            xx = ix + int(dx)
            m = finite & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if np.any(m):
                band_hit[m] |= sat[yy[m], xx[m]]
        sat_any |= band_hit
        sat_all &= band_hit
        has_band |= finite

    sat_hit = (has_band & sat_all) if require_all_bands else sat_any
    n_removed = int(sat_hit.sum())
    if n_removed > 0:
        logger.info("Saturation-cut removed %d/%d injected Gaia sources", n_removed, len(gaia_df))
    sat_x = x[sat_hit] if n_removed > 0 else np.array([])
    sat_y = y[sat_hit] if n_removed > 0 else np.array([])
    kept = gaia_df.loc[~sat_hit].copy().reset_index(drop=True)
    return kept, sat_x, sat_y


def augment_catalog_with_gaia(
    *,
    cfg: dict,
    state: dict,
) -> pd.DataFrame:
    """Inject Gaia synphot sources into the input catalog for ZP calibration.

    Returns the augmented catalog (also saved to disk). Updates state in-place.
    """
    zp_cfg = cfg["zp"]
    outputs = cfg["outputs"]
    inputs = cfg["inputs"]
    zp_ref = float(cfg["image_scaling"]["zp_ref"])

    gaia_csv_path = inputs.get("gaiaxp_synphot_csv")
    if gaia_csv_path is None or not Path(gaia_csv_path).exists():
        logger.warning("GaiaXP synphot CSV not found (%s); skipping augmentation", gaia_csv_path)
        return state["input_catalog"]

    input_catalog: pd.DataFrame = state["input_catalog"]
    image_dict: dict = state["image_dict"]
    white: np.ndarray = state["white"]

    gaia_mag_min = float(zp_cfg.get("gaia_mag_min", 11.0))
    gaia_mag_max = float(zp_cfg.get("gaia_mag_max", 18.0))
    match_radius = float(zp_cfg.get("match_radius_arcsec", 1.0))
    min_box_size = int(zp_cfg.get("min_box_size_pix", 1000))

    gaia = pd.read_csv(gaia_csv_path, dtype={"source_id": str})
    logger.info("Loaded GaiaXP synphot: %d sources", len(gaia))

    for c in ("ra", "dec", "source_id", "phot_g_mean_mag"):
        if c not in gaia.columns:
            raise ValueError(f"GaiaXP CSV missing required column: {c}")

    gaia_filt = gaia[
        (gaia["phot_g_mean_mag"] >= gaia_mag_min)
        & (gaia["phot_g_mean_mag"] <= gaia_mag_max)
    ].copy()
    logger.info("After mag cut [%.1f, %.1f]: %d sources", gaia_mag_min, gaia_mag_max, len(gaia_filt))

    wcs = next(iter(image_dict.values()))["wcs"]
    H, W = white.shape

    col_map = {str(c).strip().lower(): str(c) for c in input_catalog.columns}
    ra_col = col_map.get("ra", "RA")
    dec_col = col_map.get("dec", "DEC")
    id_col = col_map.get("id", "ID") if "id" in col_map else None

    cat_ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    cat_dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)

    x_cat, y_cat = wcs.all_world2pix(cat_ra, cat_dec, 0)

    _excl = input_catalog.get("excluded_any", pd.Series(False, index=input_catalog.index))
    _active_mask = ~_excl.fillna(False).astype(bool).to_numpy()
    x0_box, x1_box, y0_box, y1_box = _compute_gaia_selection_box(
        x_sources=x_cat[_active_mask], y_sources=y_cat[_active_mask],
        min_box_size=min_box_size, img_h=H, img_w=W,
    )
    logger.info("Gaia selection box (pix): x=[%d,%d], y=[%d,%d], size=%dx%d",
                x0_box, x1_box, y0_box, y1_box, x1_box - x0_box, y1_box - y0_box)

    gaia_ra = gaia_filt["ra"].to_numpy(dtype=float)
    gaia_dec = gaia_filt["dec"].to_numpy(dtype=float)
    x_gaia, y_gaia = wcs.all_world2pix(gaia_ra, gaia_dec, 0)

    in_box = (
        np.isfinite(x_gaia) & np.isfinite(y_gaia)
        & (x_gaia >= x0_box) & (x_gaia < x1_box)
        & (y_gaia >= y0_box) & (y_gaia < y1_box)
    )
    gaia_box = gaia_filt.loc[in_box].copy().reset_index(drop=True)
    x_gaia_box = x_gaia[in_box]
    y_gaia_box = y_gaia[in_box]
    logger.info("Gaia sources in box: %d", len(gaia_box))

    cat_sc = SkyCoord(ra=cat_ra * u.deg, dec=cat_dec * u.deg)
    gaia_sc = SkyCoord(ra=gaia_box["ra"].to_numpy() * u.deg, dec=gaia_box["dec"].to_numpy() * u.deg)

    if len(gaia_sc) == 0 or len(cat_sc) == 0:
        logger.warning("No Gaia sources in box or empty input catalog; skipping augmentation")
        input_catalog = input_catalog.copy()
        input_catalog["gaia_source_id"] = ""
        state["input_catalog"] = input_catalog
        return input_catalog

    idx_gaia, sep2d, _ = cat_sc.match_to_catalog_sky(gaia_sc)
    sep_arcsec = sep2d.to(u.arcsec).value
    matched_mask = sep_arcsec <= match_radius

    aug_cat = input_catalog.copy().reset_index(drop=True)
    gaia_sid_arr = [""] * len(aug_cat)
    for i in range(len(aug_cat)):
        if matched_mask[i]:
            gaia_sid_arr[i] = str(gaia_box.iloc[idx_gaia[i]]["source_id"])
    aug_cat["gaia_source_id"] = gaia_sid_arr

    bandnames = sorted(image_dict.keys())

    n_filled = 0
    for i in range(len(aug_cat)):
        if not matched_mask[i]:
            continue
        g = gaia_box.iloc[idx_gaia[i]]
        for bn in bandnames:
            flux_col = f"FLUX_{bn}"
            if flux_col not in aug_cat.columns:
                continue
            val = aug_cat.at[i, flux_col]
            if pd.isna(val) or val == 0:
                mag_col = f"mag_{bn}"
                if mag_col in g.index and np.isfinite(g[mag_col]):
                    aug_cat.at[i, flux_col] = float(10.0 ** ((zp_ref - float(g[mag_col])) / 2.5))
                    n_filled += 1
    if n_filled > 0:
        logger.info("Backfilled %d missing FLUX values for matched sources from Gaia synphot", n_filled)

    inject_gaia = bool(zp_cfg.get("inject_gaia_sources", True))

    matched_gaia_ids = set(v for v in gaia_sid_arr if v)
    new_gaia = gaia_box[~gaia_box["source_id"].isin(matched_gaia_ids)].copy().reset_index(drop=True)
    x_gaia_sat = np.array([])
    y_gaia_sat = np.array([])
    new_rows: list[dict] = []

    if inject_gaia:
        sat_cut_cfg = cfg.get("source_saturation_cut", {})
        if bool(sat_cut_cfg.get("enabled", True)) and len(new_gaia) > 0:
            sat_radius = float(sat_cut_cfg.get("radius_pix", 20.0))
            require_all = bool(sat_cut_cfg.get("require_all_bands", False))
            new_gaia, x_gaia_sat, y_gaia_sat = _filter_gaia_saturation(
                new_gaia, image_dict=image_dict, wcs=wcs,
                radius_pix=sat_radius, require_all_bands=require_all,
            )

        logger.info("Original sources matched to Gaia: %d | New Gaia sources to inject: %d",
                    int(matched_mask.sum()), len(new_gaia))
        for _, g in new_gaia.iterrows():
            row: dict[str, Any] = {}
            if id_col:
                row[id_col] = f"gaia_{g['source_id']}"
            row[ra_col] = float(g["ra"])
            row[dec_col] = float(g["dec"])
            type_col = col_map.get("type", "TYPE")
            row[type_col] = "STAR"
            row["gaia_source_id"] = str(g["source_id"])

            for bn in bandnames:
                mag_col = f"mag_{bn}"
                flux_col_name = f"FLUX_{bn}"
                if mag_col in g.index and np.isfinite(g[mag_col]):
                    row[flux_col_name] = float(10.0 ** ((zp_ref - float(g[mag_col])) / 2.5))
                else:
                    row[flux_col_name] = 0.0
            new_rows.append(row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            aug_cat = pd.concat([aug_cat, new_df], ignore_index=True)
    else:
        logger.info("Original sources matched to Gaia: %d | Gaia injection disabled (inject_gaia_sources=false); "
                    "%d unmatched Gaia sources skipped", int(matched_mask.sum()), len(new_gaia))

    zp_dir = Path(outputs["work_dir"]) / "ZP"
    zp_dir.mkdir(parents=True, exist_ok=True)

    out_name = Path(inputs["input_catalog"]).stem + "_with_Gaia.csv"
    out_path = zp_dir / out_name
    aug_cat.to_csv(out_path, index=False)
    logger.info("Wrote augmented catalog: %s (%d rows, %d original + %d new Gaia)",
                out_path, len(aug_cat), len(input_catalog), len(new_rows))

    if new_rows:
        new_ra = np.array([r[ra_col] for r in new_rows], dtype=float)
        new_dec = np.array([r[dec_col] for r in new_rows], dtype=float)
        x_new_gaia, y_new_gaia = wcs.all_world2pix(new_ra, new_dec, 0)
    else:
        x_new_gaia, y_new_gaia = np.array([]), np.array([])

    matched_orig_indices = np.where(matched_mask)[0]
    x_matched = x_cat[matched_orig_indices] if len(matched_orig_indices) else np.array([])
    y_matched = y_cat[matched_orig_indices] if len(matched_orig_indices) else np.array([])

    _excl_crop = input_catalog["excluded_crop"].fillna(False).to_numpy(dtype=bool) if "excluded_crop" in input_catalog.columns else None
    _excl_sat = input_catalog["excluded_saturation"].fillna(False).to_numpy(dtype=bool) if "excluded_saturation" in input_catalog.columns else None

    plot_dpi = int(cfg.get("plotting", {}).get("dpi", 150))
    _save_augmentation_overlay(
        white=white, wcs=wcs,
        x_orig=x_cat, y_orig=y_cat,
        x_gaia_new=x_new_gaia, y_gaia_new=y_new_gaia,
        x_gaia_matched=x_matched, y_gaia_matched=y_matched,
        x_gaia_sat=x_gaia_sat, y_gaia_sat=y_gaia_sat,
        excluded_crop=_excl_crop,
        excluded_saturation=_excl_sat,
        bbox=(x0_box, x1_box, y0_box, y1_box),
        outpath=zp_dir / "gaia_augmentation_overlay.png",
        dpi=plot_dpi,
    )

    state["input_catalog"] = aug_cat
    state["augmented_catalog_path"] = out_path
    return aug_cat


# ---------------------------------------------------------------------------
#  ZP computation
# ---------------------------------------------------------------------------

def _sigma_clip_mad(z: np.ndarray, clip_sigma: float, max_iters: int) -> np.ndarray:
    """Return boolean mask of points to keep after iterative MAD-based clipping."""
    keep = np.isfinite(z)
    for _ in range(int(max_iters)):
        zz = z[keep]
        if len(zz) < 3:
            break
        med = float(np.median(zz))
        mad = float(np.median(np.abs(zz - med)))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break
        prev = keep.copy()
        keep = np.isfinite(z) & (np.abs(z - med) <= clip_sigma * sigma)
        if np.array_equal(keep, prev):
            break
    return keep


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of values."""
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cumw = np.cumsum(sw)
    half = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, half)
    return float(sv[min(idx, len(sv) - 1)])


def _save_zp_band_plot(
    *,
    df: pd.DataFrame,
    band: str,
    zp_med: float,
    zp_mad_sig: float,
    outpath: Path,
    dpi: int = 150,
    nonconv_rows: list[dict] | None = None,
) -> None:
    """Save per-band ZP diagnostic plot (kept + clipped + non-converged)."""
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    if np.isfinite(zp_mad_sig) and zp_mad_sig > 0:
        shade = plt.axhspan(zp_med - zp_mad_sig, zp_med + zp_mad_sig,
                             color="0.8", alpha=0.35)
    else:
        shade = None

    if nonconv_rows:
        df_nc = pd.DataFrame(nonconv_rows)
        plt.scatter(
            df_nc["gaiaxp_mag"].to_numpy(), df_nc["zp"].to_numpy(),
            s=25, marker="d", c="0.6", alpha=0.6,
            label=f"non-converged (N={len(df_nc)})",
        )

    d1 = df.loc[df["is_clipped"]]
    if len(d1):
        plt.scatter(
            d1["gaiaxp_mag"].to_numpy(), d1["zp"].to_numpy(),
            s=35, marker="x", c="tab:red", alpha=0.9,
            label=f"clipped (N={len(d1)})",
        )

    d0 = df.loc[df["is_kept"]]
    if len(d0):
        plt.errorbar(
            d0["gaiaxp_mag"].to_numpy(), d0["zp"].to_numpy(),
            yerr=d0["zp_err"].to_numpy(),
            fmt="o", ms=4, mfc="none", mec="k", ecolor="0.4",
            elinewidth=0.7, capsize=0, alpha=0.9,
            label=f"kept (N={len(d0)})",
        )

    med_line = plt.axhline(zp_med, color="tab:red", lw=1.5,
                            label=f"median={zp_med:.3f} $\\pm$ {zp_mad_sig:.3f}")
    if shade is not None:
        shade.set_label("median $\\pm$ MAD$\\sigma$")

    plt.xlabel("GaiaXP synphot AB mag")
    plt.ylabel("ZP (mag)")
    plt.title(band)

    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ["non-converged", "clipped", "kept", "median", "MAD"]
    ordered_hl = []
    for prefix in desired_order:
        for h, l in zip(handles, labels):
            if l.startswith(prefix):
                ordered_hl.append((h, l))
    for h, l in zip(handles, labels):
        if not any(l == ol for _, ol in ordered_hl):
            ordered_hl.append((h, l))
    if ordered_hl:
        plt.legend([h for h, _ in ordered_hl], [l for _, l in ordered_hl],
                   loc="best", fontsize=8, framealpha=0.7)
    else:
        plt.legend(loc="best", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def compute_zp(
    *,
    cfg: dict,
    merged_catalog_path: Path,
) -> Path:
    """Compute per-band ZP from Gaia-matched stars and apply AB magnitudes.

    Overwrites the merged catalog with additional MAG/MAGERR columns.
    Returns the path to the updated catalog.
    """
    zp_cfg = cfg["zp"]
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    clip_sigma = float(zp_cfg.get("clip_sigma", 3.0))
    clip_max_iters = int(zp_cfg.get("clip_max_iters", 10))

    gaia_csv_path = inputs.get("gaiaxp_synphot_csv")
    if gaia_csv_path is None or not Path(gaia_csv_path).exists():
        logger.warning("GaiaXP synphot CSV not found; skipping ZP computation")
        return merged_catalog_path

    cat = pd.read_csv(merged_catalog_path, dtype={"gaia_source_id": str})
    gaia = pd.read_csv(gaia_csv_path, dtype={"source_id": str})

    if "gaia_source_id" not in cat.columns:
        logger.warning("No gaia_source_id column in merged catalog; skipping ZP computation")
        return merged_catalog_path

    cat["gaia_source_id"] = cat["gaia_source_id"].fillna("").astype(str).str.strip()
    has_gaia = (cat["gaia_source_id"] != "") & (cat["gaia_source_id"] != "nan")

    is_multi_band = "opt_converged" in cat.columns
    gaia_nonconv: pd.DataFrame | None = None
    if is_multi_band:
        converged = cat["opt_converged"].fillna(False).astype(bool)
        n_gaia_total = int(has_gaia.sum())
        n_not_converged = int((has_gaia & ~converged).sum())
        gaia_sources = cat[has_gaia & converged].copy()
        gaia_nonconv = cat[has_gaia & ~converged].copy() if n_not_converged > 0 else None
        logger.info("Gaia-matched sources for ZP: %d (excluded %d non-converged out of %d)",
                    len(gaia_sources), n_not_converged, n_gaia_total)
    else:
        gaia_sources = cat[has_gaia].copy()
        logger.info("Gaia-matched sources for ZP: %d (per-band convergence filtering applied below)",
                    len(gaia_sources))

    if len(gaia_sources) == 0:
        logger.warning("No Gaia-matched sources found; skipping ZP computation")
        return merged_catalog_path

    gaia_lookup = gaia.drop_duplicates(subset="source_id").set_index("source_id")

    flux_pat = re.compile(r"^FLUX_(.+)_fit$")
    flux_cols = [c for c in cat.columns if flux_pat.match(c)]
    bandnames = sorted(flux_pat.match(c).group(1) for c in flux_cols)
    logger.info("Bands for ZP: %s", ", ".join(bandnames))

    zp_dir = Path(outputs["work_dir"]) / "ZP"
    zp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    all_zp_rows: list[pd.DataFrame] = []

    for band in bandnames:
        flux_col = f"FLUX_{band}_fit"
        ferr_col = f"FLUXERR_{band}_fit"
        gaia_mag_col = f"mag_{band}"

        if flux_col not in gaia_sources.columns:
            continue
        if gaia_mag_col not in gaia_lookup.columns:
            logger.info("[%s] no mag_%s in GaiaXP CSV; skip", band, band)
            continue

        band_conv_col = f"opt_converged_{band}"
        if not is_multi_band and band_conv_col in gaia_sources.columns:
            band_converged = gaia_sources[band_conv_col].fillna(False).astype(bool)
        else:
            band_converged = pd.Series(True, index=gaia_sources.index)

        def _build_zp_rows(src_df):
            rows = []
            for _, src in src_df.iterrows():
                gid = str(src["gaia_source_id"]).strip()
                if not gid or gid == "nan" or gid not in gaia_lookup.index:
                    continue
                gaia_row = gaia_lookup.loc[gid]
                gaia_mag = float(gaia_row[gaia_mag_col]) if np.isfinite(gaia_row[gaia_mag_col]) else np.nan
                flux = float(src[flux_col]) if flux_col in src.index else np.nan
                fluxerr = float(src[ferr_col]) if ferr_col in src.index and np.isfinite(src[ferr_col]) else np.nan
                if not (np.isfinite(gaia_mag) and np.isfinite(flux) and flux > 0):
                    continue
                zp_i = gaia_mag + 2.5 * np.log10(flux)
                zp_err_i = K_MAG * (fluxerr / flux) if (np.isfinite(fluxerr) and fluxerr > 0) else np.nan
                rows.append({
                    "gaia_source_id": gid, "gaiaxp_mag": gaia_mag,
                    "flux": flux, "fluxerr": fluxerr, "zp": zp_i, "zp_err": zp_err_i,
                })
            return rows

        rows_band = _build_zp_rows(gaia_sources.loc[band_converged])
        rows_nonconv = _build_zp_rows(gaia_sources.loc[~band_converged])
        if is_multi_band and gaia_nonconv is not None:
            rows_nonconv.extend(_build_zp_rows(gaia_nonconv))
        n_nonconv = len(rows_nonconv)
        if n_nonconv > 0:
            logger.info("[%s] excluded %d non-converged Gaia sources from ZP", band, n_nonconv)

        if not rows_band:
            logger.info("[%s] no valid ZP rows", band)
            continue

        df = pd.DataFrame(rows_band)

        keep = _sigma_clip_mad(df["zp"].to_numpy(), clip_sigma, clip_max_iters)
        df["is_kept"] = keep
        df["is_clipped"] = ~keep

        df_keep = df.loc[df["is_kept"]].copy()
        if len(df_keep) == 0:
            logger.info("[%s] all points clipped; skip", band)
            continue

        zp_vals = df_keep["zp"].to_numpy()
        zp_errs = df_keep["zp_err"].to_numpy()
        valid_err = np.isfinite(zp_errs) & (zp_errs > 0)

        if np.any(valid_err):
            weights = np.where(valid_err, 1.0 / (zp_errs ** 2), 0.0)
            if weights.sum() > 0:
                zp_med = _weighted_median(zp_vals[valid_err], weights[valid_err])
            else:
                zp_med = float(np.median(zp_vals))
        else:
            zp_med = float(np.median(zp_vals))

        zp_mad_sig = float(1.4826 * np.median(np.abs(zp_vals - zp_med)))

        summary_rows.append({
            "band": band,
            "n_raw": int(len(df)),
            "n_kept": int(len(df_keep)),
            "n_clipped": int((~keep).sum()),
            "n_nonconverged": n_nonconv,
            "clip_sigma": float(clip_sigma),
            "zp_median": zp_med,
            "zp_err_mad": zp_mad_sig,
        })

        df.insert(0, "band", band)
        all_zp_rows.append(df)

        _save_zp_band_plot(
            df=df, band=band, zp_med=zp_med, zp_mad_sig=zp_mad_sig,
            outpath=zp_dir / f"zp_vs_mag__{band}.png",
            dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
            nonconv_rows=rows_nonconv if rows_nonconv else None,
        )
        logger.info("[%s] ZP=%.4f +/- %.4f (MAD) | kept=%d nonconv=%d clipped=%d",
                     band, zp_med, zp_mad_sig, int(len(df_keep)), n_nonconv, int((~keep).sum()))

    if all_zp_rows:
        zp_long = pd.concat(all_zp_rows, ignore_index=True)
        zp_long.to_csv(zp_dir / "zp_all_bands__per_object.csv", index=False)
        logger.info("Wrote per-object ZP: %s", zp_dir / "zp_all_bands__per_object.csv")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(zp_dir / "zp_summary.csv", index=False)
        logger.info("Wrote ZP summary: %s", zp_dir / "zp_summary.csv")

        zp_map = {row["band"]: (row["zp_median"], row["zp_err_mad"]) for row in summary_rows}

        mag_updates: dict[str, np.ndarray] = {}
        for band in bandnames:
            mag_col = f"MAG_{band}_fit"
            magerr_col = f"MAGERR_{band}_fit"
            if band not in zp_map:
                mag_updates[mag_col] = np.full(len(cat), np.nan, dtype=float)
                mag_updates[magerr_col] = np.full(len(cat), np.nan, dtype=float)
                continue

            zp, zp_err = zp_map[band]
            flux_col = f"FLUX_{band}_fit"
            ferr_col = f"FLUXERR_{band}_fit"

            flux = pd.to_numeric(cat[flux_col], errors="coerce").to_numpy(dtype=float) if flux_col in cat.columns else np.full(len(cat), np.nan)
            ferr = pd.to_numeric(cat[ferr_col], errors="coerce").to_numpy(dtype=float) if ferr_col in cat.columns else np.full(len(cat), np.nan)

            ok = np.isfinite(flux) & (flux > 0)
            mag = np.full(len(cat), np.nan, dtype=float)
            magerr = np.full(len(cat), np.nan, dtype=float)

            mag[ok] = zp - 2.5 * np.log10(flux[ok])

            ok_err = ok & np.isfinite(ferr) & (ferr > 0)
            magerr_flux = np.full(len(cat), np.nan, dtype=float)
            magerr_flux[ok_err] = K_MAG * (ferr[ok_err] / flux[ok_err])
            magerr[ok_err] = np.sqrt(magerr_flux[ok_err] ** 2 + float(zp_err) ** 2)

            mag_updates[mag_col] = mag
            mag_updates[magerr_col] = magerr

        if mag_updates:
            replace_cols = [c for c in mag_updates if c in cat.columns]
            if replace_cols:
                cat = cat.drop(columns=replace_cols)
            cat = pd.concat([cat, pd.DataFrame(mag_updates, index=cat.index)], axis=1)

        cat.to_csv(merged_catalog_path, index=False)
        logger.info("Updated merged catalog with MAG/MAGERR columns: %s", merged_catalog_path)
    else:
        logger.warning("No bands produced ZP results; catalog unchanged")

    return merged_catalog_path
