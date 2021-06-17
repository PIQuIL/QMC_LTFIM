#!/bin/bash
for M in $(seq 10000 10000 100000)
do
  for p in $(seq 1)
  do
    for delta in $(seq 1.0 0.04 1.2)
    do
      X="$M|$delta"
      echo $X
      sbatch -J "$X" --export="nY=16,delta=$delta,M=$M,p=$p" submit_rydberg
    done
  done
  sleep 0.5s
done

