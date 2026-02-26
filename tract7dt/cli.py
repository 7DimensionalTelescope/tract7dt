from __future__ import annotations

import argparse
import datetime
import logging
import shutil
from pathlib import Path

from .config import load_config, write_sample_config
from .logging_utils import setup_logging
from .pipeline import (
    build_epsf,
    build_patch_inputs,
    build_patches,
    load_inputs,
    merge_results,
    run_patch_subprocesses,
    run_pipeline,
)
from .sample_data import download_sample_data


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True, help="Path to YAML config")


def _load_and_augment(cfg: dict) -> dict:
    """Load inputs and conditionally augment with Gaia sources.

    Mirrors the first two stages of ``run_pipeline`` so that step commands
    produce the same catalog state as a full run.
    """
    state = load_inputs(cfg)
    if bool(cfg.get("zp", {}).get("enabled", False)):
        from .zp import augment_catalog_with_gaia
        augment_catalog_with_gaia(cfg=cfg, state=state)
    return state


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    log = logging.getLogger("tract7dt.cli")

    from . import __version__

    ap = argparse.ArgumentParser(prog="tract7dt")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bump = sub.add_parser("dump-config", help="Write the latest sample config file")
    ap_bump.add_argument("--out", default="sample_config.yaml", help="Output path for sample config")
    ap_bump.add_argument("--force", action="store_true", help="Overwrite if output file exists")

    ap_dl = sub.add_parser("download-sample", help="Download sample image data")
    ap_dl.add_argument(
        "--dir",
        dest="download_dir",
        default=None,
        help="Target dataset directory path (default: ./<original-sample-dir-name>)",
    )
    ap_dl.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target dataset directory",
    )

    ap_run = sub.add_parser("run", help="Run full pipeline")
    _add_common(ap_run)

    ap_epsf = sub.add_parser("run-epsf", help="Build ePSF only")
    _add_common(ap_epsf)

    ap_patches = sub.add_parser("build-patches", help="Build patch definitions only")
    _add_common(ap_patches)

    ap_inputs = sub.add_parser("build-patch-inputs", help="Build patch inputs only")
    _add_common(ap_inputs)

    ap_run_p = sub.add_parser("run-patches", help="Run patch subprocesses only")
    _add_common(ap_run_p)

    ap_merge = sub.add_parser("merge", help="Merge patch outputs only")
    _add_common(ap_merge)

    ap_zp = sub.add_parser("compute-zp", help="Compute zero-point calibration from merged catalog")
    _add_common(ap_zp)

    args = ap.parse_args(argv)

    if args.cmd == "dump-config":
        out = write_sample_config(args.out, overwrite=bool(args.force))
        log.info("Wrote sample config: %s", out)
        return 0

    if args.cmd == "download-sample":
        if args.download_dir:
            target_dir = Path(args.download_dir).expanduser()
            parent_dir = target_dir.parent
        else:
            target_dir = None
            parent_dir = Path.cwd()
        result = download_sample_data(
            path_to_download=parent_dir,
            target_dataset_dir=target_dir,
            overwrite=bool(args.force),
        )
        log.info("Sample data is ready at: %s", result.dataset_dir)
        log.info("Generated image list: %s", result.image_list_file)
        return 0

    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("file")
    if isinstance(log_file, str) and log_file.strip().lower() == "auto":
        log_cfg["file"] = Path(cfg["outputs"]["work_dir"]) / f"{args.cmd}.log"
    setup_logging(log_cfg, force=True)

    work_dir = Path(cfg["outputs"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_src = cfg.get("config_path")
    if cfg_src and Path(cfg_src).exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = work_dir / f"config_used_{ts}.yaml"
        shutil.copy2(str(cfg_src), str(snap))
        log.info("Config snapshot saved: %s", snap)

    if args.cmd == "run":
        run_pipeline(cfg)
    elif args.cmd == "run-epsf":
        state = _load_and_augment(cfg)
        build_epsf(cfg, state)
    elif args.cmd == "build-patches":
        state = _load_and_augment(cfg)
        build_patches(cfg, state)
    elif args.cmd == "build-patch-inputs":
        state = _load_and_augment(cfg)
        build_patches(cfg, state)
        build_patch_inputs(cfg, state)
    elif args.cmd == "run-patches":
        run_patch_subprocesses(cfg)
    elif args.cmd == "merge":
        state = _load_and_augment(cfg)
        merge_results(cfg, state)
    elif args.cmd == "compute-zp":
        from .zp import compute_zp
        merged_path = Path(cfg["outputs"]["final_catalog"])
        if not merged_path.exists():
            raise SystemExit(f"Merged catalog not found: {merged_path}. Run 'merge' first.")
        compute_zp(cfg=cfg, merged_catalog_path=merged_path)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    return 0
