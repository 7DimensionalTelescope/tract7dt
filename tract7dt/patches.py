from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.patches")


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


def _grid_bounds_in_bbox(x0: int, x1: int, y0: int, y1: int, ngrid: int, r: int, c: int):
    if not (0 <= r < ngrid and 0 <= c < ngrid):
        raise ValueError("r/c out of range")
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("bbox has non-positive size")

    x_step = width // ngrid
    y_step = height // ngrid
    if x_step <= 0 or y_step <= 0:
        raise ValueError("ngrid too fine for bbox size")

    bx0 = x0 + c * x_step
    bx1 = x1 if c == ngrid - 1 else x0 + (c + 1) * x_step
    by0 = y0 + r * y_step
    by1 = y1 if r == ngrid - 1 else y0 + (r + 1) * y_step
    return bx0, bx1, by0, by1


def build_patches(
    *,
    epsf_ngrid: int,
    patch_ngrid: int,
    final_psf_size: int,
    halo_pix_min: int,
    image_shape: tuple[int, int],
    out_dir: Path,
    active_epsf_tags: set[str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    HALO_PIX = max(int((final_psf_size - 1) // 2 + 2), int(halo_pix_min))

    ny, nx = image_shape
    patches = []

    for er in range(epsf_ngrid):
        for ec in range(epsf_ngrid):
            epsf_tag = f"r{er:02d}_c{ec:02d}"
            if active_epsf_tags is not None and epsf_tag not in active_epsf_tags:
                continue
            ex0, ex1, ey0, ey1 = _grid_bounds(nx, ny, epsf_ngrid, er, ec)

            for pr in range(patch_ngrid):
                for pc in range(patch_ngrid):
                    x0b, x1b, y0b, y1b = _grid_bounds_in_bbox(ex0, ex1, ey0, ey1, patch_ngrid, pr, pc)

                    x0r = max(0, x0b - HALO_PIX)
                    x1r = min(nx, x1b + HALO_PIX)
                    y0r = max(0, y0b - HALO_PIX)
                    y1r = min(ny, y1b + HALO_PIX)

                    patch_tag = f"r{er:02d}_c{ec:02d}_pr{pr:02d}_pc{pc:02d}"

                    patches.append(
                        dict(
                            epsf_r=int(er),
                            epsf_c=int(ec),
                            patch_r=int(pr),
                            patch_c=int(pc),
                            epsf_tag=epsf_tag,
                            patch_tag=patch_tag,
                            x0_base=int(x0b),
                            x1_base=int(x1b),
                            y0_base=int(y0b),
                            y1_base=int(y1b),
                            x0_roi=int(x0r),
                            x1_roi=int(x1r),
                            y0_roi=int(y0r),
                            y1_roi=int(y1r),
                            halo_pix=int(HALO_PIX),
                        )
                    )

    patch_df = pd.DataFrame(patches)
    patch_csv = out_dir / "patches.csv"
    patch_json = out_dir / "patches.json"
    patch_df.to_csv(patch_csv, index=False)
    patch_json.write_text(json.dumps(patches, indent=2))

    logger.info("Image shape: %s", (ny, nx))
    logger.info("EPSF_NGRID: %d | PATCH_NGRID: %d | HALO_PIX: %d", epsf_ngrid, patch_ngrid, HALO_PIX)
    if active_epsf_tags is not None:
        logger.info("Active ePSF cells: %d/%d", len(active_epsf_tags), epsf_ngrid * epsf_ngrid)
    logger.info("Total patches: %d", len(patches))
    logger.info("Wrote: %s", patch_csv)
    logger.info("Wrote: %s", patch_json)

    return patch_json
