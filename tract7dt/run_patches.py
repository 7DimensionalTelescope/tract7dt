"""
Orchestrate patch-level Tractor runs via subprocesses.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .logging_utils import setup_logging

logger = logging.getLogger("tract7dt.run_patches")


def _subprocess_env(*, threads_per_process: int, package_root: Path) -> dict:
    env = os.environ.copy()
    t = str(int(max(1, threads_per_process)))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[k] = t
    pkg_path = str(package_root)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pkg_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pkg_path
    return env


def _patch_tag_from_input_path(p: Path) -> str:
    name = p.name
    m = re.match(r"^(.*)\.pkl\.gz$", name)
    if m:
        return m.group(1)
    return p.stem


def _should_run(p: Path, outdir_base: Path, resume: bool) -> bool:
    if not resume:
        return True
    tag = _patch_tag_from_input_path(p)
    out_csv = outdir_base / tag / f"{tag}_cat_fit.csv"
    return not out_csv.exists()


def _run_one(
    *,
    patch_input: Path,
    runner_module: str,
    epsf_root: Path,
    outdir_base: Path,
    python_exe: str,
    threads_per_process: int,
    resume: bool,
    extra_args: list[str],
    package_root: Path,
) -> dict:
    tag = _patch_tag_from_input_path(patch_input)
    patch_out = outdir_base / tag
    patch_out.mkdir(parents=True, exist_ok=True)

    if resume and (not _should_run(patch_input, outdir_base, resume=True)):
        return {"patch": tag, "status": "skipped", "returncode": 0, "sec": 0.0}

    cmd = [
        python_exe,
        "-m",
        runner_module,
        "--inputs",
        str(patch_input),
        "--epsf-root",
        str(epsf_root),
        "--outdir",
        str(outdir_base),
    ]
    cmd.extend(extra_args)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            env=_subprocess_env(threads_per_process=threads_per_process, package_root=package_root),
        )
        dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        runner_log = patch_out / f"{tag}.runner.log"
        runner_log.write_text("CMD: " + " ".join(cmd) + "\n\n" + f"Runner exception: {e}\n")
        return {"patch": tag, "status": "fail", "returncode": -1, "sec": float(dt), "runner_log": str(runner_log)}

    runner_log = patch_out / f"{tag}.runner.log"
    runner_log.write_text(
        "CMD: " + " ".join(cmd) + "\n\n"
        + "--- STDOUT ---\n" + (proc.stdout or "") + "\n\n"
        + "--- STDERR ---\n" + (proc.stderr or "") + "\n"
    )

    return {
        "patch": tag,
        "status": "ok" if proc.returncode == 0 else "fail",
        "returncode": int(proc.returncode),
        "sec": float(dt),
        "runner_log": str(runner_log),
    }


def run_subprocesses(
    *,
    patch_input_dir: Path,
    epsf_root: Path,
    outdir: Path,
    python_exe: str,
    max_workers: int,
    threads_per_process: int,
    resume: bool,
    extra_args: list[str],
) -> Path:
    runner_module = "tract7dt.tractor_patch_runner"
    package_root = Path(__file__).resolve().parents[1]
    outdir.mkdir(parents=True, exist_ok=True)

    if not patch_input_dir.exists():
        raise SystemExit(f"Missing patch input dir: {patch_input_dir}")

    inputs_all = sorted(patch_input_dir.glob("*.pkl.gz"))
    inputs = [p for p in inputs_all if _should_run(p, outdir, resume=bool(resume))]

    logger.info("Found inputs: %d | to run: %d | max_workers=%d", len(inputs_all), len(inputs), int(max_workers))

    results: list[dict] = []
    failures: list[dict] = []
    tstart = time.time()
    use_inline_progress = sys.stderr.isatty()

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

    with cf.ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        fut_to_input = {
            ex.submit(
                _run_one,
                patch_input=p,
                runner_module=runner_module,
                epsf_root=epsf_root,
                outdir_base=outdir,
                python_exe=str(python_exe),
                threads_per_process=int(threads_per_process),
                resume=bool(resume),
                extra_args=extra_args,
                package_root=package_root,
            ): p
            for p in inputs
        }
        pending = set(fut_to_input.keys())
        n = len(pending)
        done = 0
        ok = 0
        skipped = 0
        last_non_tty_log = 0.0

        def _emit_progress(force_log: bool = False) -> None:
            nonlocal last_non_tty_log
            elapsed = max(1e-6, time.time() - tstart)
            rate = done / elapsed
            running = len(pending)
            remaining = max(0, n - done)
            eta = float("inf") if rate <= 0 else remaining / rate
            msg = (
                f"Run patches [{_progress_bar(done, n)}] "
                f"{done}/{n} ok={ok} skipped={skipped} fail={len(failures)} "
                f"running={running} rate={rate:.2f}/s eta={_format_eta(eta)}"
            )
            if use_inline_progress:
                sys.stderr.write("\r" + msg)
                sys.stderr.flush()
            else:
                now = time.time()
                if force_log or (now - last_non_tty_log) >= 15.0:
                    logger.info(msg)
                    last_non_tty_log = now

        if n == 0:
            _emit_progress(force_log=True)
        else:
            _emit_progress(force_log=True)
            while pending:
                done_now, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
                if not done_now:
                    _emit_progress(force_log=False)
                    continue
                for fut in done_now:
                    res = fut.result()
                    results.append(res)
                    done += 1
                    status = str(res.get("status", ""))
                    if status == "ok":
                        ok += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failures.append(res)
                _emit_progress(force_log=True)

        if use_inline_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

    df = pd.DataFrame(results)
    summary_csv = outdir / "run_summary.csv"
    df.to_csv(summary_csv, index=False)
    logger.info(
        "DONE. total: %d ok: %s fail: %d",
        len(results),
        int((df["status"] == "ok").sum()) if "status" in df.columns else "?",
        len(failures),
    )
    logger.info("Wrote: %s", summary_csv)
    if failures:
        logger.warning("First few failures:")
        for r in failures[:10]:
            logger.warning(
                "patch=%s rc=%s log=%s",
                r.get("patch"),
                r.get("returncode"),
                r.get("runner_log"),
            )
    return summary_csv


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-input-dir", required=True, help="Directory containing patch input .pkl.gz files")
    ap.add_argument("--epsf-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--threads-per-process", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--python", default=os.sys.executable)
    ap.add_argument("--no-cutouts", action="store_true")
    ap.add_argument("--no-patch-overview", action="store_true")
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
    args = ap.parse_args(argv)

    extra_args: list[str] = [
        "--cutout-size",
        str(int(args.cutout_size)),
        "--cutout-start",
        str(int(args.cutout_start)),
    "--cutout-data-p-lo",
    str(float(args.cutout_data_p_lo)),
    "--cutout-data-p-hi",
    str(float(args.cutout_data_p_hi)),
    "--cutout-model-p-lo",
    str(float(args.cutout_model_p_lo)),
    "--cutout-model-p-hi",
    str(float(args.cutout_model_p_hi)),
    "--cutout-resid-p-abs",
    str(float(args.cutout_resid_p_abs)),
        "--gal-model",
        str(args.gal_model),
        "--n-opt-iters",
        str(int(args.n_opt_iters)),
        "--dlnp-stop",
        str(float(args.dlnp_stop)),
        "--flag-maxiter-margin",
        str(int(args.flag_maxiter_margin)),
    "--r-ap",
    str(float(args.r_ap)),
    "--eps-flux",
    str(float(args.eps_flux)),
    "--psf-model",
    str(args.psf_model),
    "--psf-fallback-model",
    str(args.psf_fallback_model),
    "--min-epsf-nstars-for-use",
    str(int(args.min_epsf_nstars_for_use)),
    "--ncircular-gaussian-n",
    str(int(args.ncircular_gaussian_n)),
    "--hybrid-psf-n",
    str(int(args.hybrid_psf_n)),
    "--gaussian-mixture-n",
    str(int(args.gaussian_mixture_n)),
    "--sersic-n-init",
    str(float(args.sersic_n_init)),
    "--re-fallback-pix",
    str(float(args.re_fallback_pix)),
        "--fallback-default-fwhm-pix",
        str(float(args.fallback_default_fwhm_pix)),
        "--fallback-stamp-radius-pix",
        str(float(args.fallback_stamp_radius_pix)),
        "--moffat-beta",
        str(float(args.moffat_beta)),
        "--moffat-radius-pix",
        str(float(args.moffat_radius_pix)),
    ]
    if args.cutout_max_sources is not None:
        extra_args.extend(["--cutout-max-sources", str(int(args.cutout_max_sources))])
    if args.no_cutouts:
        extra_args.append("--no-cutouts")
    if args.no_patch_overview:
        extra_args.append("--no-patch-overview")
    run_subprocesses(
        patch_input_dir=Path(args.patch_input_dir),
        epsf_root=Path(args.epsf_root),
        outdir=Path(args.outdir),
        python_exe=str(args.python),
        max_workers=int(args.max_workers),
        threads_per_process=int(args.threads_per_process),
        resume=bool(args.resume),
        extra_args=extra_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
