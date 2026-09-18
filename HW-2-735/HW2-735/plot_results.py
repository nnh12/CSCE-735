#!/usr/bin/env python3
#
# Analyze benchmark results and demonstrate speedup for the best list sizes.
#
# For each k it computes, at every q:
#   speedup(p)     = T(k, q=0) / T(k, q)
#   efficiency(p)  = speedup(p) / p
#
# It picks the two k values that achieve the largest speedup and produces:
#   results_summary.csv  - timing, speedup, efficiency for every run (all k)
#   timing.png           - wall-clock time vs threads (best two k)
#   speedup.png          - speedup vs threads + ideal line (best two k)
#   efficiency.png       - efficiency vs threads + ideal line (best two k)
#
# Usage: ./plot_results.py [results.csv]

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MARKERS = ["o", "s", "^", "D"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            t = r.get("time_sec", "")
            if t and t.lower() not in ("", "na", "failed"):
                rows.append(r)
    return rows


def analyze(rows):
    ks = sorted({int(r["k"]) for r in rows})
    per_k = {}
    for k in ks:
        t1 = None
        for r in rows:
            if int(r["k"]) == k and int(r["q"]) == 0:
                t1 = float(r["time_sec"])
                break
        if t1 is None:
            continue
        stats = []
        for r in rows:
            if int(r["k"]) != k:
                continue
            p = 1 << int(r["q"])
            t = float(r["time_sec"])
            sp = t1 / t
            eff = sp / p
            stats.append((int(r["q"]), p, t, sp, eff))
        per_k[k] = (t1, sorted(stats, key=lambda s: s[1]))
    return per_k


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results_1.csv"
    rows = load(path)
    if not rows:
        print(f"No valid runs in {path}")
        sys.exit(1)

    per_k = analyze(rows)
    if not per_k:
        print("No k with a q=0 baseline run found")
        sys.exit(1)

    ranked = sorted(per_k, key=lambda k: max(s[3] for s in per_k[k][1]), reverse=True)
    best = ranked[:2]

    with open("results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "q", "threads", "list_size", "time_sec",
                    "speedup", "efficiency"])
        for k in ranked:
            t1, stats = per_k[k]
            for (q, p, t, sp, eff) in stats:
                size = 1 << k
                w.writerow([k, q, p, size, f"{t:.6f}", f"{sp:.4f}", f"{eff:.4f}"])

    print(f"{'k':>3} {'q':>3} {'threads':>7} {'time(sec)':>12} {'speedup':>10} {'efficiency':>12}")
    print("-" * 55)
    for k in ranked:
        t1, stats = per_k[k]
        for (q, p, t, sp, eff) in stats:
            print(f"{k:>3} {q:>3} {p:>7} {t:>12.6f} {sp:>10.4f} {eff:>12.4f}")
        print("-" * 55)

    for k in best:
        t1, stats = per_k[k]
        pe, bspot = max(((s[3], s) for s in stats), key=lambda x: x[0])
        print(f"k={k}: base {t1:.4f}s, peak speedup {pe:.2f}x "
              f"at {bspot[1]} threads (T={bspot[2]:.4f}s)")
    print(f"Best two k values by max speedup: {best}")
    print("Full table written to results_summary.csv")

    colors = plt.cm.tab10(np.linspace(0, 1, len(best)))
    for title, idx, ylabel, logx, logy, fname in (
        ("Wall-clock Time vs Threads", 2, "Time (sec)", True, True, "timing.png"),
        ("Speedup vs Threads", 3, "Speedup", True, False, "speedup.png"),
        ("Efficiency vs Threads", 4, "Efficiency", True, False, "efficiency.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, color, marker in zip(best, colors, MARKERS):
            t1, stats = per_k[k]
            xs = [s[1] for s in stats]
            ys = [s[idx] for s in stats]
            ax.plot(xs, ys, marker=marker, color=color, label=f"k={k}")
        if idx == 3:
            ax.plot(xs, xs, "--", color="gray", label="ideal (S=p)")
        elif idx == 4:
            ax.axhline(1.0, linestyle="--", color="gray", label="ideal (E=1)")
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(t) for t in xs])
        ax.set_xlabel("Number of threads (p)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"Wrote {fname}")


if __name__ == "__main__":
    main()