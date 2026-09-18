#!/usr/bin/env python3
"""
Benchmark the parallel OpenMP Strassen executable.

Usage:
  python3 bench_strassen.py                    # defaults below
  python3 bench_strassen.py --exe ./strassen_omp.exe
  python3 bench_strassen.py --kmin 10 --kmax 14 --q 3 4 5 6 7 8 9 --reps 3
  python3 bench_strassen.py --plot            # also write graphs

For every matrix size 2**k (k in [--kmin, --kmax]) and every leaf size
2**q, it times the executable with 1 thread (T1) and with all available
cores (Tp), then computes:

    speedup(p)   = T1 / Tp
    efficiency(p)= speedup(p) / p    where p = number of threads

All values are written to a CSV (default strassen_speedup.csv) so you can
plot the results later on any machine. With --plot it also writes
speedup_vs_leaf.png and efficiency_vs_leaf.png (x = leaf size 2^q, one
line per matrix size k) showing which leaf sizes give the best speedup.

The executable is invoked as:
    ./strassen_omp.exe <log2(matrix_size)> <log2(leaf_size)>
"""

import argparse
import csv
import os
import re
import subprocess
import sys

DEFAULT_EXE = "./strassen_omp.exe"

TIME_RE = re.compile(r"Strassen's \(s\) =\s+([\d.]+)")


def available_cores():
    """Return the total number of CPUs on this machine."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run_once(exe, k, q, threads, timeout):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    try:
        proc = subprocess.run([exe, str(k), str(q)],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True, env=env,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    m = TIME_RE.search(proc.stdout)
    if not m:
        return None
    return float(m.group(1))


def best_time(exe, k, q, threads, reps, timeout):
    """Run reps times and return the fastest time (or None)."""
    best = None
    for _ in range(reps):
        t = run_once(exe, k, q, threads, timeout)
        if t is not None and (best is None or t < best):
            best = t
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exe", default=os.environ.get("STRASSEN_EXE", DEFAULT_EXE),
                   help=f"path to executable (default: {DEFAULT_EXE})")
    p.add_argument("--kmin", type=int, default=10,
                   help="smallest log2(matrix_size) (default: 10)")
    p.add_argument("--kmax", type=int, default=14,
                   help="largest log2(matrix_size) (default: 14)")
    p.add_argument("--q", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9],
                   help="log2(leaf_size) values to try (default: 3 4 5 6 7 8 9)")
    p.add_argument("--threads", type=int, default=0,
                   help="number of threads for the parallel run, 0 = all cores "
                        "(default: 0)")
    p.add_argument("--reps", type=int, default=3,
                   help="runs per configuration; fastest is kept (default: 3)")
    p.add_argument("--timeout", type=int, default=3600,
                   help="timeout in seconds per single run (default: 3600)")
    p.add_argument("--out", default="strassen_speedup.csv",
                   help="output CSV (default: strassen_speedup.csv)")
    p.add_argument("--plot", action="store_true",
                   help="also write a speedup plot (PNG)")
    args = p.parse_args()

    if not os.path.isfile(args.exe):
        sys.exit(f"Executable not found: {args.exe}")

    p_threads = args.threads if args.threads > 0 else available_cores()

    qs = [q for q in sorted(set(args.q)) if 2 ** q >= 1]
    ks = list(range(args.kmin, args.kmax + 1))

    rows = []
    for k in ks:
        size = 1 << k
        for q in qs:
            leaf = 1 << q
            if leaf > size:
                continue  # program would clamp leaf to matrix size anyway
            # 1-thread baseline
            t1 = best_time(args.exe, k, q, 1, args.reps, args.timeout)
            if t1 is None:
                print(f"k={k:2d} q={q} leaf={leaf:5d}  FAILED (1 thread)")
                continue
            # parallel run on all available cores
            tp = best_time(args.exe, k, q, p_threads, args.reps, args.timeout)
            if tp is None:
                print(f"k={k:2d} q={q} leaf={leaf:5d}  FAILED ({p_threads} threads)")
                continue
            speedup = t1 / tp
            efficiency = speedup / p_threads
            rows.append({
                "k": k, "matrix_size": size, "q": q, "leaf_size": leaf,
                "threads": p_threads, "t1_sec": f"{t1:.6f}",
                "tp_sec": f"{tp:.6f}", "speedup": f"{speedup:.4f}",
                "efficiency": f"{efficiency:.4f}",
            })
            print(f"k={k:2d} q={q} leaf={leaf:5d} | T1={t1:8.4f}s "
                  f"Tp={tp:8.4f}s | speedup={speedup:6.2f}x "
                  f"eff={efficiency:6.2f}")

    if not rows:
        sys.exit("No successful runs recorded.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "k", "matrix_size", "q", "leaf_size", "threads",
            "t1_sec", "tp_sec", "speedup", "efficiency"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Best leaf size per k (max speedup, later k wins ties)
    best = {}
    for r in rows:
        k = int(r["k"])
        sp = float(r["speedup"])
        if k not in best or sp >= best[k][1]:
            best[k] = (r, sp)
    print("\nBest leaf size per matrix size (by max speedup):")
    print(f"{'k':>3} {'matrix':>7} {'leaf(q)':>11} {'threads':>7} {'speedup':>9} {'eff':>7}")
    for k in sorted(best):
        r, sp = best[k]
        print(f"{k:>3} {r['matrix_size']:>7} {r['leaf_size']:>6}(q={r['q']}) "
              f"{r['threads']:>7} {sp:>9.4f} {float(r['efficiency']):>7.4f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        by_k = {}
        for r in rows:
            by_k.setdefault(int(r["k"]), []).append(r)
        for ylabel, idx, fname in (
            ("Speedup", "speedup", "speedup_vs_leaf.png"),
            ("Efficiency", "efficiency", "efficiency_vs_leaf.png"),
        ):
            fig, ax = plt.subplots(figsize=(8, 6))
            leaf_ticks = sorted({int(r["leaf_size"]) for r in rows})
            for k in sorted(by_k):
                pts = by_k[k]
                xs = [int(r["leaf_size"]) for r in pts]
                ys = [float(r[idx]) for r in pts]
                ax.plot(xs, ys, marker="o", label=f"k={k} (n={1<<k})")
                b = max(pts, key=lambda r: float(r["speedup"]))
                if idx == "speedup":
                    ax.annotate(f"{float(b['speedup']):.2f}x@q={b['q']}",
                                xy=(int(b["leaf_size"]), float(b["speedup"])),
                                textcoords="offset points", xytext=(5, 6),
                                fontsize=8)
            ax.set_xscale("log", base=2)
            pn = p_threads
            if idx == "speedup":
                ax.plot([max(1, min(leaf_ticks)), max(leaf_ticks)],
                        [pn, pn], "--", color="gray",
                        label=f"ideal (S={pn})")
            else:
                ax.axhline(1.0, linestyle="--", color="gray",
                           label="ideal (E=1)")
            ax.set_xticks(leaf_ticks)
            ax.set_xticklabels([str(s) for s in leaf_ticks])
            ax.set_xlabel("Leaf matrix size (2^q)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"OpenMP Strassen {ylabel} vs leaf size "
                         f"({p_threads} threads)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            print(f"Wrote {fname}")


if __name__ == "__main__":
    main()