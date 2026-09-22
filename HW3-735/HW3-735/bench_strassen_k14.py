#!/usr/bin/env python3
"""
Run the Strassen benchmark for k = 14 only (matrix size 2^14 = 16384).

Thin wrapper around bench_strassen.py that forces --kmin 14 --kmax 14
while accepting every other bench_strassen.py option.

Examples:
  python3 bench_strassen_k14.py                          # full leaf sweep, reps 2
  python3 bench_strassen_k14.py --q 6 7 --reps 1         # quick k=14 run
  python3 bench_strassen_k14.py --plot --out k14.csv     # also plots
"""

import sys

import bench_strassen

FORCED = [("--kmin", "14"), ("--kmax", "14")]

args = sys.argv[1:]

# Force the k range to 14 unless the user already gave it explicitly.
for flag, value in FORCED:
    if flag not in args:
        args = args + [flag, value]

sys.argv = ["bench_strassen_k14.py"] + args
bench_strassen.main()