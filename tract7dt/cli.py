from __future__ import annotations

import argparse
import logging
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


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    log = logging.getLogger("tract7dt.cli")

    ap = argparse.ArgumentParser(prog="tract7dt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bump = sub.add_parser("dump-config", help="Write the latest sample config file")
    ap_bump.add_argument("--out", default="sample_config.yaml", help="Output path for sample config")
    ap_bump.add_argument("--force", action="store_true", help="Overwrite if output file exists")

    ap_dl = sub.add_parser("download-sample-data", help="Download sample image data")
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

    args = ap.parse_args(argv)

    if args.cmd == "dump-config":
        out = write_sample_config(args.out, overwrite=bool(args.force))
        log.info("Wrote sample config: %s", out)
        return 0

    if args.cmd == "download-sample-data":
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
    setup_logging(cfg.get("logging", {}), force=True)

    if args.cmd == "run":
        run_pipeline(cfg)
    elif args.cmd == "run-epsf":
        state = load_inputs(cfg)
        build_epsf(cfg, state)
    elif args.cmd == "build-patches":
        state = load_inputs(cfg)
        build_patches(cfg, state)
    elif args.cmd == "build-patch-inputs":
        state = load_inputs(cfg)
        build_patches(cfg, state)
        build_patch_inputs(cfg, state)
    elif args.cmd == "run-patches":
        run_patch_subprocesses(cfg)
    elif args.cmd == "merge":
        merge_results(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    return 0
