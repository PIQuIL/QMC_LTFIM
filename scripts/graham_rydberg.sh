#!/bin/bash
for M in $(seq 0 2000 40000)
do
  for p in $(seq 0.0 0.1 0.9)
  do
    X="M=$M,p=$p"
    sbatch -J "$X" --export="$X" submit_rydberg
  done
done
