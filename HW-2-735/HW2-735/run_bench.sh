#!/bin/bash
# Run ./sort_list.exe <k> <q> for every combination of
#   k = 12, 20, 28 and q = 0, 1, 2, 4, 6, 8, 10
# Results are appended to results.csv

EXE=./sort_list.exe
OUT_FILE=results.csv

if [ ! -x "$EXE" ]; then
    echo "Executable $EXE not found. Compile first with:"
    echo "    icx -o sort_list.exe sort_list.c -lpthread"
    exit 1
fi

echo "k,q,threads,list_size,time_sec,qsort_time_sec,error" > "$OUT_FILE"

for k in 12 20 28; do
    for q in 0 1 2 4 6 8 10; do
        out="$($EXE "$k" "$q" 2>&1)"
        printf "k=%s q=%s -> %s\n" "$k" "$q" "$out"

        list_size=$(printf '%s\n' "$out" | sed -n 's/.*List Size = \([0-9]*\).*/\1/p')
        threads=$(printf '%s\n' "$out" | sed -n 's/.*Threads = \([0-9]*\).*/\1/p')
        err=$(printf '%s\n' "$out" | sed -n 's/.*error = \([0-9]*\).*/\1/p')
        time_sec=$(printf '%s\n' "$out" | sed -n 's/.*time (sec) = *\([0-9.]*\).*/\1/p')
        qsort_sec=$(printf '%s\n' "$out" | sed -n 's/.*qsort_time = *\([0-9.]*\).*/\1/p')

        if [ -z "$time_sec" ]; then
            printf "%s,%s,FAILED,%s\n" "$k" "$q" "list_size unsupported or run failed" >> "$OUT_FILE"
        else
            printf "%s,%s,%s,%s,%s,%s,%s\n" "$k" "$q" "$threads" "$list_size" "$time_sec" "$qsort_sec" "$err" >> "$OUT_FILE"
        fi
    done
done

echo "Results saved to $OUT_FILE"