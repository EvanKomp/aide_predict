#!/usr/bin/env python
"""
rerun_all_scoring.py — Re-run new_05b_bnn2_score.py on every CV run dir.

Discovers all directories under --results-root that contain both
``pairwise_predictions.csv`` and ``train_metadata.json`` (i.e. completed
``new_05_bnn2_train.py`` runs) and re-runs the scoring/plotting pipeline on
each. The existing ``--output-subdir`` (default ``scoring``) is wiped before
each run so stale figures from previous schemas never linger.

Usage:
  # Re-score every run found under results/new_05_bnn2 (default root)
  python rerun_all_scoring.py

  # Custom root, custom output subdir, restrict to two modes
  python rerun_all_scoring.py \
      --results-root /path/to/results \
      --output-subdir scoring_v3 \
      --modes formaldehyde nearest

  # Parallelize across 4 worker processes
  python rerun_all_scoring.py --jobs 4

  # Just list what would be done
  python rerun_all_scoring.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SCORE_SCRIPT = SCRIPT_DIR / "new_05b_bnn2_score.py"
DEFAULT_RESULTS_ROOT = (
    SCRIPT_DIR.parent.parent / "results" / "new_05_bnn2"
).resolve()


def discover_run_dirs(root: Path) -> List[Path]:
    """Run dirs are directories that contain both pairwise_predictions.csv
    and train_metadata.json (the two artifacts the scoring script needs)."""
    runs = []
    for pw in root.rglob("pairwise_predictions.csv"):
        rd = pw.parent
        if (rd / "train_metadata.json").exists():
            runs.append(rd)
    return sorted(set(runs))


def wipe_output_subdir(run_dir: Path, subdir: str) -> bool:
    """Remove ``<run_dir>/<subdir>`` if it exists. Returns True if removed."""
    out = run_dir / subdir
    if not out.exists():
        return False
    if not out.is_dir():
        raise RuntimeError(
            f"Refusing to wipe non-directory output target: {out}")
    shutil.rmtree(out)
    return True


def run_one(
    run_dir: Path,
    output_subdir: str,
    modes: Optional[List[str]],
    extra_args: List[str],
    python_exec: str,
) -> Tuple[Path, int, float, str]:
    """Wipe stale output, invoke the scoring script. Returns (dir, rc, dt, log_tail)."""
    wipe_output_subdir(run_dir, output_subdir)
    cmd = [
        python_exec, str(SCORE_SCRIPT),
        "--run-dir", str(run_dir),
        "--output-subdir", output_subdir,
    ]
    if modes:
        cmd += ["--modes", *modes]
    cmd += extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    # Tail the combined stream so failures are visible in the summary.
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return run_dir, proc.returncode, dt, tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                   help=f"Root directory to search for runs (default: {DEFAULT_RESULTS_ROOT})")
    p.add_argument("--output-subdir", type=str, default="scoring",
                   help="Subdir under each run dir for scoring outputs (default: scoring)")
    p.add_argument("--modes", type=str, nargs="*", default=None,
                   help="Pass through to scoring script (default: all 4 modes)")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Parallel worker processes (default: 1 = sequential)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be done, but don't wipe or re-score")
    p.add_argument("--python", type=str, default=sys.executable,
                   help=f"Python executable for subprocesses (default: {sys.executable})")
    p.add_argument("score_args", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to new_05b_bnn2_score.py "
                        "(prefix with -- to separate, e.g. "
                        "`-- --distance-weighted-temperature 2.0`)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not SCORE_SCRIPT.exists():
        print(f"ERROR: scoring script not found: {SCORE_SCRIPT}", file=sys.stderr)
        return 2

    root = args.results_root.resolve()
    if not root.is_dir():
        print(f"ERROR: --results-root not a directory: {root}", file=sys.stderr)
        return 2

    runs = discover_run_dirs(root)
    if not runs:
        print(f"No CV runs found under {root}", file=sys.stderr)
        return 1

    # Strip the leading "--" if argparse left it in REMAINDER
    extra = list(args.score_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    print(f"Found {len(runs)} run dir(s) under {root}:")
    for rd in runs:
        out = rd / args.output_subdir
        flag = " (will wipe)" if out.exists() else ""
        print(f"  {rd.relative_to(root)}{flag}")
    print(f"\nScoring script:  {SCORE_SCRIPT}")
    print(f"Output subdir:   {args.output_subdir}")
    print(f"Modes:           {args.modes or 'all'}")
    print(f"Extra args:      {extra or '(none)'}")
    print(f"Workers:         {args.jobs}")
    if args.dry_run:
        print("\n[dry-run] Not executing.")
        return 0

    print()
    results: List[Tuple[Path, int, float, str]] = []
    t_start = time.time()

    if args.jobs <= 1:
        for rd in runs:
            print(f"── {rd.relative_to(root)} ──")
            res = run_one(rd, args.output_subdir, args.modes, extra, args.python)
            results.append(res)
            status = "OK" if res[1] == 0 else f"FAIL(rc={res[1]})"
            print(f"  {status} in {res[2]:.1f}s")
            if res[1] != 0:
                print("  tail:")
                for line in res[3].splitlines():
                    print(f"    {line}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(run_one, rd, args.output_subdir, args.modes, extra,
                          args.python): rd for rd in runs
            }
            for fut in as_completed(futures):
                rd = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = (rd, -1, 0.0, f"worker exception: {e!r}")
                results.append(res)
                status = "OK" if res[1] == 0 else f"FAIL(rc={res[1]})"
                print(f"[{status}] {rd.relative_to(root)}  ({res[2]:.1f}s)")
                if res[1] != 0:
                    for line in res[3].splitlines():
                        print(f"    {line}")

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    n_ok = sum(1 for r in results if r[1] == 0)
    n_fail = len(results) - n_ok
    print(f"  succeeded: {n_ok}/{len(results)}")
    if n_fail:
        print(f"  failed:    {n_fail}")
        for rd, rc, dt, tail in results:
            if rc != 0:
                print(f"    - {rd.relative_to(root)}  (rc={rc}, {dt:.1f}s)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
