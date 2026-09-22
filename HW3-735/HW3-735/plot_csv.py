#!/usr/bin/env python3
"""
Plot speedup and efficiency from a bench_strassen.py results CSV.

Reads a CSV with columns: k, matrix_size, q, leaf_size, threads,
t1_sec, tp_sec, speedup, efficiency

Outputs:
  speedup_vs_leaf.png     - speedup vs leaf size (2^q), one line per matrix size k
  efficiency_vs_leaf.png  - efficiency vs leaf size (2^q), one line per matrix size k

Usage:
  python3 plot_csv.py                       # reads strassen_speedup.csv
  python3 plot_csv.py results.csv           # custom file
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DEFAULT = "strassen_speedup.csv"


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                sp = float(r["speedup"])
                eff = float(r["efficiency"])
                t1 = float(r["t1_sec"])
                tp = float(r["tp_sec"])
            except (ValueError, KeyError):
                continue  # skip incomplete/empty rows
            rows.append({
                "k": int(r["k"]),
                "q": int(r["q"]),
                "leaf_size": int(r["leaf_size"]),
                "threads": int(r["threads"]),
                "t1_sec": t1,
                "tp_sec": tp,
                "speedup": sp,
                "efficiency": eff,
            })
    return rows


def build(rows):
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)
    threads = rows[0]["threads"] if rows else 1
    return by_k, threads


def make_plot(by_k, threads, ykey, ylabel, ideal_label, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    leaf_ticks = sorted({r["leaf_size"] for k in by_k for r in by_k[k]})
    for k in sorted(by_k):
        pts = sorted(by_k[k], key=lambda r: r["leaf_size"])
        xs = [r["leaf_size"] for r in pts]
        ys = [r[ykey] for r in pts]
        ax.plot(xs, ys, marker="o", label=f"k={k} (n={1 << k})")
        best = max(pts, key=lambda r: (r["speedup"], r["leaf_size"]))
        if ykey == "speedup":
            ax.annotate(f"{best['speedup']:.2f}x@q={best['q']}",
                        xy=(best["leaf_size"], best[ykey]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaf_ticks)
    ax.set_xticklabels([str(s) for s in leaf_ticks])
    if ykey == "speedup":
        ax.plot([max(1, min(leaf_ticks)), max(leaf_ticks)],
                [threads, threads], "--", color="gray",
                label=ideal_label)
    else:
        ax.axhline(1.0, linestyle="--", color="gray", label=ideal_label)
    ax.set_xlabel("Leaf matrix size (2^q)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"OpenMP Strassen {ylabel} vs leaf size ({threads} threads)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Wrote {fname}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    rows = load_rows(path)
    if not rows:
        print(f"No valid rows in {path}")
        sys.exit(1)
    by_k, threads = build(rows)
    print(f"{len(rows)} rows, {len(by_k)} matrix sizes (k = "
          f"{min(by_k)}..{max(by_k)}), {threads} threads")

    make_plot(by_k, threads, "speedup", "Speedup",
              f"ideal (S={threads})", "speedup_vs_leaf.png")
    make_plot(by_k, threads, "efficiency", "Efficiency",
              "ideal (E=1)", "efficiency_vs_leaf.png")

    print("\nBest leaf size (max speedup) per matrix size:")
    for k in sorted(by_k):
        best = max(by_k[k], key=lambda r: (r["speedup"], r["leaf_size"]))
        print(f"  k={k:>2} n={1 << k:>5}  leaf=2^{best['q']}={best['leaf_size']:>4}  "
              f"speedup={best['speedup']:.2f}x  efficiency={best['efficiency']:.3f}")


if __name__ == "__main__":
    main()