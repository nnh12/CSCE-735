#!/usr/bin/env python3
"""
Plot the weak-scaling results produced by weak_scaling.grace_job.

Reads weak_results.csv with columns: p,n_total,Tp,T1,S,E
Writes weak_scaling.png with execution time, speedup, and efficiency.

Run with the Anaconda module (the system python3 has no matplotlib):
    module load Anaconda3/2026.07-1
    python plot_weak.py
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "weak_results.csv"
OUT = "weak_scaling.png"

rows = [r for r in csv.DictReader(open(CSV))]
p = [float(r["p"]) for r in rows]
tp = [float(r["Tp"]) for r in rows]
s = [float(r["S"]) for r in rows]
e = [float(r["E"]) for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

ax[0].plot(p, tp, "o-")
ax[0].set_ylabel("Execution time Tp (s)")

ax[1].plot(p, s, "o-", label="measured")
ax[1].plot(p, p, "k--", label="ideal")
ax[1].set_ylabel("Speedup S(p) = T1(n*p) / Tp(n)")
ax[1].legend()

ax[2].plot(p, e, "o-")
ax[2].axhline(1.0, color="k", ls="--")
ax[2].set_ylabel("Efficiency E(p) = S(p) / p")

for a in ax:
    a.set_xscale("log", base=2)
    a.set_xlabel("Number of processes p")
    a.set_xticks(p)
    a.set_xticklabels([str(int(x)) for x in p])
    a.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=200)
print("Wrote", OUT)
