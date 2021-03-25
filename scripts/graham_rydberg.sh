#!/bin/bash
for i in $(seq 0 9)
do
  delta_i=$(echo "1.01 + ($i*0.02)" | bc)
  delta_f=$(echo "$delta_i + 0.01" | bc)
  deltas=$(echo "$delta_i\n$delta_f")
  for nY in $(seq 8 8 32)
  do
    X="nY=$nY,deltas=$deltas"
    sbatch -J "nY=$nY_δ=$delta_i-$delta_f" --export="$X" submit_rydberg
    sleep 0.5s
  done
done
