#!/bin/bash
for M in $(seq 2000 2000 10000)
do
  for p in $(seq 0.4 0.1 0.5)
  do
    for delta in $(seq 1.01 0.01 1.2)
    do
      X="delta=$delta,M=$M,p=$p"
      sbatch -J "p=$p,d=$delta,M=$M" --export="$X" submit_rydberg
    done
    sleep 2s
  done
done

