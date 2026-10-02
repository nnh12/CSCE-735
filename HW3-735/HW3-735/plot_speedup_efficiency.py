#!/usr/bin/env python3
"""
Plot speedup and efficiency from a Strassen benchmark CSV.

Reads a CSV with columns:
    k, matrix_size, q, leaf_size, threads, t1_sec, tp_sec, speedup, efficiency

Produces two figures, each with one line per matrix size k:
    speedup_vs_leaf.png     - y = speedup,    x = leaf matrix size 2^q
    efficiency_vs_leaf.png  - y = efficiency, x = leaf matrix size 2^q

Usage:
    python3 plot_speedup_efficiency.py                       # strassen_speedup_final.csv
    python3 plot_speedup_efficiency.py other.csv             # custom CSV
    python3 plot_speedup_efficiency.py --outdir figs other.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DEFAULT = "strassen_speedup_final.csv"


def load_rows(path):
    """Read the CSV, skipping rows with missing/failed timings."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "k": int(r["k"]),
                    "q": int(r["q"]),
                    "leaf_size": int(r["leaf_size"]),
                    "threads": int(r["threads"]),
                    "speedup": float(r["speedup"]),
                    "efficiency": float(r["efficiency"]),
                })
            except (ValueError, KeyError):
                continue  # incomplete row (e.g. a run that failed)
    return rows


def make_plot(rows, ykey, ylabel, ideal, fname):
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)

    threads = rows[0]["threads"] if rows else 1
    leaf_ticks = sorted({r["leaf_size"] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in sorted(by_k):
        pts = sorted(by_k[k], key=lambda r: r["leaf_size"])
        xs = [r["leaf_size"] for r in pts]
        ys = [r[ykey] for r in pts]
        ax.plot(xs, ys, marker="o", label=f"k={k} (n={1 << k})")

        # annotate the best point on each k-curve (by speedup)
        best = max(pts, key=lambda r: r["speedup"])
        ax.annotate(f"{best['speedup']:.2f}x @ 2^{best['q']}",
                    xy=(best["leaf_size"], best[ykey]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    # x axis: leaf size on a base-2 log scale with explicit 2^q ticks
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaf_ticks)
    ax.set_xticklabels([str(s) for s in leaf_ticks])
    ax.set_xlabel("Leaf matrix size (2^q)")

    # reference line
    if ykey == "speedup":
        ax.plot([min(leaf_ticks), max(leaf_ticks)], [threads, threads],
                "--", color="gray", label=f"ideal (S={threads})")
    else:
        ax.axhline(1.0, linestyle="--", color="gray", label="ideal (E=1)")

    ax.set_ylabel(ylabel)
    ax.set_title(f"OpenMP Strassen {ylabel} vs leaf size ({threads} threads)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Wrote {fname}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=CSV_DEFAULT,
                   help=f"input CSV (default: {CSV_DEFAULT})")
    p.add_argument("--outdir", default=".", help="output directory (default: .)")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No valid rows in {args.csv}")

    by_k = sorted({r["k"] for r in rows})
    print(f"{len(rows)} rows, {len(by_k)} matrix sizes "
          f"(k = {min(by_k)}..{max(by_k)})")

    make_plot(rows, "speedup", "Speedup", "ideal",
              os.path.join(args.outdir, "speedup_vs_leaf.png"))
    make_plot(rows, "efficiency", "Efficiency", "ideal",
              os.path.join(args.outdir, "efficiency_vs_leaf.png"))

    print("\nBest leaf size (max speedup) per matrix size:")
    for k in by_k:
        pts = [r for r in rows if r["k"] == k]
        best = max(pts, key=lambda r: r["speedup"])
        print(f"  k={k:>2}  n={1 << k:>5}  leaf=2^{best['q']}={best['leaf_size']:>4}  "
              f"speedup={best['speedup']:.2f}x  efficiency={best['efficiency']:.3f}")


if __name__ == "__main__":
    main()
