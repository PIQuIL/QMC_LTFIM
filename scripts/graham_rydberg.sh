#!/bin/bash
for M in $(seq 1)
do
  for p in $(seq 1)
  do
    for delta in $(seq -3.0 0.05 3.0)
    do
      M=100000
      X="$delta|$p"
      echo $X
      sbatch -J "$X" --export="nY=16,delta=$delta,M=$M,p=$p" submit_rydberg
    done
  done
  sleep 0.5s
done




