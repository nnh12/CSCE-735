#!/bin/bash

# Keep total number of MPI processes fixed at 64
for tpn in 1 2 4 8 16 32 64
do
    nodes=$((64 / tpn))

    echo "Submitting: nodes=$nodes, ntasks-per-node=$tpn"

    sbatch \
        --nodes=$nodes \
        --ntasks-per-node=$tpn \
        --job-name=pi_tpn_$tpn \
        --output=output_tpn_${tpn}.%j \
        5.4.grace_job
done
