"""
Merge patch-level Tractor results back into a single catalog.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import setup_logging

logger = logging.getLogger("tract7dt.merge")


def _pick_key_cols(df: pd.DataFrame) -> list[str]:
    if "ID" in df.columns:
        return ["ID"]
    ra = "RA" if "RA" in df.columns else ("ra" if "ra" in df.columns else None)
    dec = "DEC" if "DEC" in df.columns else ("dec" if "dec" in df.columns else None)
    if ra and dec:
        return [ra, dec]
    raise ValueError("Could not find merge key columns. Need 'ID' or ('RA','DEC').")


def merge_catalogs(
    *,
    input_catalog: Path,
    patch_outdir: Path,
    out_path: Path,
    pattern: str,
    wcs_fits: Path | None,
    exclusion_flags: pd.DataFrame | None = None,
) -> Path:
    if not input_catalog.exists():
        raise SystemExit(f"Missing input catalog: {input_catalog}")
    if not patch_outdir.exists():
        raise SystemExit(f"Missing patch outdir: {patch_outdir}")

    base = pd.read_csv(input_catalog)
    key_cols = _pick_key_cols(base)
    if exclusion_flags is not None and len(exclusion_flags) > 0:
        flag_cols = [c for c in ("excluded_crop", "excluded_saturation") if c in exclusion_flags.columns]
        if flag_cols:
            flags = exclusion_flags.copy()
            for c in flag_cols:
                flags[c] = flags[c].fillna(False).astype(bool)
            flags = flags.groupby(key_cols, as_index=False, dropna=False)[flag_cols].max()
            base = base.merge(flags, how="left", on=key_cols)
            for c in flag_cols:
                base[c] = base[c].fillna(False).astype(bool)
            if "excluded_crop" not in base.columns:
                base["excluded_crop"] = False
            if "excluded_saturation" not in base.columns:
                base["excluded_saturation"] = False
            base["excluded_any"] = base["excluded_crop"] | base["excluded_saturation"]
            base["excluded_reason"] = np.select(
                [
                    base["excluded_crop"] & base["excluded_saturation"],
                    base["excluded_crop"],
                    base["excluded_saturation"],
                ],
                [
                    "crop+saturation",
                    "crop",
                    "saturation",
                ],
                default="",
            )

    csvs = sorted(patch_outdir.glob(f"**/{pattern}"))
    if len(csvs) == 0:
        raise SystemExit(f"No patch result CSVs found under {patch_outdir} with pattern {pattern}")

    rows: list[pd.DataFrame] = []
    for p in csvs:
        df = pd.read_csv(p)
        legacy_force = ("x_fit_white", "y_fit_white")
        fit_cols = [c for c in df.columns if c not in base.columns or c in legacy_force]
        keep_cols = list(dict.fromkeys(key_cols + fit_cols))
        keep_cols = [c for c in keep_cols if c in df.columns]
        rows.append(df[keep_cols])

    fit = pd.concat(rows, ignore_index=True)

    if fit.duplicated(subset=key_cols).any():
        dup = fit.loc[fit.duplicated(subset=key_cols, keep=False), key_cols].head(10)
        raise ValueError(f"Duplicate keys in patch results (showing up to 10):\n{dup}")

    merged = base.merge(fit, how="left", on=key_cols, suffixes=("", "_fitdup"))

    dup_cols = [c for c in merged.columns if c.endswith("_fitdup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    if wcs_fits:
        want = ("RA_fit" in merged.columns) and ("DEC_fit" in merged.columns)
        have_xy = ("x_pix_white_fit" in merged.columns) and ("y_pix_white_fit" in merged.columns)
        if not want:
            merged["RA_fit"] = np.nan
            merged["DEC_fit"] = np.nan
            want = True
        if want and have_xy:
            try:
                from astropy.io import fits  # type: ignore
                from astropy.wcs import WCS  # type: ignore

                with fits.open(str(wcs_fits), memmap=True) as hdul:
                    wcs = WCS(hdul[0].header)

                xw = pd.to_numeric(merged["x_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
                yw = pd.to_numeric(merged["y_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(xw) & np.isfinite(yw)
                ra = merged["RA_fit"].to_numpy(dtype=float, copy=True)
                dec = merged["DEC_fit"].to_numpy(dtype=float, copy=True)
                fill = ok & (~np.isfinite(ra) | ~np.isfinite(dec))
                if np.any(fill):
                    r, d = wcs.all_pix2world(xw[fill], yw[fill], 0)
                    ra[fill] = np.asarray(r, dtype=float)
                    dec[fill] = np.asarray(d, dtype=float)
                    merged["RA_fit"] = ra
                    merged["DEC_fit"] = dec
                    logger.info("Filled RA_fit/DEC_fit using WCS from: %s", wcs_fits)
            except Exception as e:
                logger.warning("failed to compute RA_fit/DEC_fit in merge: %s", str(e))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info("Wrote: %s", out_path)
    logger.info("rows: %d cols: %d", len(merged), len(merged.columns))
    if "excluded_any" in merged.columns:
        n_crop = int(merged.get("excluded_crop", pd.Series(False, index=merged.index)).astype(bool).sum())
        n_sat = int(merged.get("excluded_saturation", pd.Series(False, index=merged.index)).astype(bool).sum())
        n_any = int(merged["excluded_any"].astype(bool).sum())
        logger.info("exclusions: crop=%d saturation=%d any=%d", n_crop, n_sat, n_any)
    missing = int(merged[key_cols[0]].isna().sum()) if len(key_cols) == 1 else int(np.any(merged[key_cols].isna(), axis=1).sum())
    logger.info("merge-missing (left join rows without fit): %d", missing)
    return out_path


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-catalog", required=True, help="Input catalog CSV used to build patch inputs")
    ap.add_argument("--patch-outdir", required=True, help="Base output directory containing per-patch results")
    ap.add_argument("--out", required=True, help="Output merged CSV path")
    ap.add_argument("--pattern", default="*_cat_fit.csv", help="Glob pattern for patch result CSVs")
    ap.add_argument("--wcs-fits", default=None, help="Optional FITS path for RA_fit/DEC_fit")
    args = ap.parse_args(argv)

    merge_catalogs(
        input_catalog=Path(args.input_catalog),
        patch_outdir=Path(args.patch_outdir),
        out_path=Path(args.out),
        pattern=str(args.pattern),
        wcs_fits=Path(args.wcs_fits) if args.wcs_fits else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
