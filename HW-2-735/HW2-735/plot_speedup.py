#!/usr/bin/env python3
#
# Plot speedup and efficiency from results.csv produced by run_bench.sh.
#
# speedup(p)  = T(q=0) / T(p)
# efficiency(p) = speedup(p) / p
#
# Outputs:
#   speedup_eff.csv   - table with speedup and efficiency for every run
#   speedup.png       - speedup vs number of threads (one line per k)
#   efficiency.png    - efficiency vs number of threads (one line per k)

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "results_1.csv"
TPL = [("0", "o"), ("1", "s"), ("2", "^")]


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["time_sec"] and r["time_sec"].lower() not in ("", "na", "failed"):
                rows.append({k: r[k] for k in r})
    return rows


def main():
    rows = load_results(CSV_FILE)
    if not rows:
        print(f"No valid runs found in {CSV_FILE}. Run run_bench.sh first.")
        sys.exit(1)

    ks = sorted({int(r["k"]) for r in rows})

    with open("speedup_eff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "q", "threads", "list_size", "time_sec",
                    "qsort_time_sec", "error", "speedup", "efficiency"])

        for k in ks:
            t1 = None
            for r in rows:
                if int(r["k"]) == k and int(r["q"]) == 0:
                    t1 = float(r["time_sec"])
                    break
            if t1 is None:
                print(f"k={k}: no q=0 (base) run, skipping")
                continue

            print(f"k={k}: base time (q=0) = {t1:.6f} s")
            for r in rows:
                if int(r["k"]) != k:
                    continue
                p = 1 << int(r["q"])
                t = float(r["time_sec"])
                speedup = t1 / t
                eff = speedup / p
                w.writerow([r["k"], r["q"], p, r["list_size"], r["time_sec"],
                            r["qsort_time_sec"], r["error"],
                            f"{speedup:.4f}", f"{eff:.4f}"])

    data = {k: [] for k in ks}
    threads = set()
    with open("speedup_eff.csv", newline="") as f:
        for r in csv.DictReader(f):
            k = int(r["k"])
            data[k].append((int(r["threads"]), float(r["speedup"]),
                            float(r["efficiency"])))
            threads.add(int(r["threads"]))
    x = sorted(threads)

    markers = {k: m for k, (_, m) in zip(ks, TPL)}
    colors = plt.cm.tab10(np.linspace(0, 1, len(ks)))

    for title, idx, ylabel, fname in (
        ("Speedup vs Threads", 1, "Speedup", "speedup.png"),
        ("Efficiency vs Threads", 2, "Efficiency", "efficiency.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        for k, color in zip(ks, colors):
            pts = sorted(data[k], key=lambda v: v[0])
            ax.plot([p[0] for p in pts], [p[idx] for p in pts],
                    marker=markers[k], color=color, label=f"k={k}")
        if idx == 1:
            ax.plot(x, x, "--", color="gray", label="ideal (S=p)")
        else:
            ax.axhline(1.0, linestyle="--", color="gray", label="ideal (E=1)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in x])
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
