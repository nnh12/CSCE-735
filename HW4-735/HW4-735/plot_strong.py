#!/usr/bin/env python3
"""
Plot the strong-scaling results produced by strong_scaling.grace_job.

Reads strong_results.csv with columns: p,n,Tp,T1,S,E
where
    N  = 20,480,000 (fixed total list size)
    n  = N / p      (local list size on each of the p processes)
    Tp = time on p processes, each sorting n elements
    T1 = time on 1 process sorting the same total N elements
    S  = T1 / Tp
    E  = S / p

Writes (PNG for viewing, PDF for the report):
    strong_time.png/.pdf        execution time Tp vs p
    strong_speedup.png/.pdf     speedup S vs p (with ideal S = p)
    strong_efficiency.png/.pdf  efficiency E vs p (with ideal E = 1)
    strong_scaling.png/.pdf     the three panels side by side

Run with the Anaconda module (the system python3 has no matplotlib):
    module load Anaconda3/2024.02-1
    python plot_strong.py
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "strong_results.csv"


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "p": int(r["p"]),
                    # accept both headers (older job script wrote n_total)
                    "n": int(r.get("n", r.get("n_total"))),
                    "Tp": float(r["Tp"]),
                    "T1": float(r["T1"]),
                    "S": float(r["S"]),
                    "E": float(r["E"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["p"])
    return rows


def log2_axis(ax, ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("Number of processes $p$ (log scale)")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)


def main():
    rows = load(CSV)
    if len(rows) < 2:
        sys.exit(f"Need at least two data rows in {CSV}")

    ps = [r["p"] for r in rows]
    tp = [r["Tp"] for r in rows]
    sp = [r["S"] for r in rows]
    ef = [r["E"] for r in rows]

    # ---- individual plots -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, tp, "o-", color="tab:blue", linewidth=2, markersize=7)
    log2_axis(ax, ps)
    ax.set_ylabel("Execution time $T_p$ (s)")
    ax.set_title("Strong Scaling: Execution Time")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("strong_time.png", dpi=200)
    fig.savefig("strong_time.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, sp, "o-", color="tab:orange", linewidth=2, markersize=7,
            label="measured")
    ax.plot(ps, ps, "k--", linewidth=1.5, label="ideal ($S=p$)")
    log2_axis(ax, ps)
    ax.set_ylabel("Speedup $S(p) = T_1(N) / T_p(N/p)$")
    ax.set_title("Strong Scaling: Speedup")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig("strong_speedup.png", dpi=200)
    fig.savefig("strong_speedup.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, ef, "o-", color="tab:green", linewidth=2, markersize=7,
            label="measured")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.5,
               label="ideal ($E=1$)")
    log2_axis(ax, ps)
    ax.set_ylabel("Efficiency $E(p) = S(p)/p$")
    ax.set_title("Strong Scaling: Efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("strong_efficiency.png", dpi=200)
    fig.savefig("strong_efficiency.pdf")
    plt.close(fig)

    # ---- combined 1x3 figure ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(ps, tp, "o-", color="tab:blue", linewidth=2, markersize=6)
    axes[0].set_ylabel("Execution time $T_p$ (s)")
    axes[0].set_title("Execution Time")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(ps, sp, "o-", color="tab:orange", linewidth=2, markersize=6,
                 label="measured")
    axes[1].plot(ps, ps, "k--", linewidth=1.5, label="ideal ($S=p$)")
    axes[1].set_ylabel("Speedup $S(p)$")
    axes[1].set_title("Speedup")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    axes[2].plot(ps, ef, "o-", color="tab:green", linewidth=2, markersize=6,
                 label="measured")
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1.5,
                    label="ideal ($E=1$)")
    axes[2].set_ylabel("Efficiency $E(p)$")
    axes[2].set_title("Efficiency")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="lower left")

    for a in axes:
        log2_axis(a, ps)

    fig.suptitle("Strong Scaling: total size $N = 20{,}480{,}000$, "
                 "$n = N/p$ per process", fontsize=12)
    fig.tight_layout()
    fig.savefig("strong_scaling.png", dpi=200)
    fig.savefig("strong_scaling.pdf")
    plt.close(fig)

    # ---- console table ----------------------------------------------------
    print(f"{'p':>4} {'n (local)':>14} {'Tp (s)':>10} {'T1 (s)':>10} "
          f"{'S(p)':>8} {'E(p)':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['p']:>4} {r['n']:>14} {r['Tp']:>10.4f} "
              f"{r['T1']:>10.4f} {r['S']:>8.4f} {r['E']:>8.4f}")

    print("\nWrote strong_time, strong_speedup, strong_efficiency, "
          "strong_scaling (.png and .pdf)")


if __name__ == "__main__":
    main()
