#!/bin/bash
for nY in $(seq 8 8 32)
do
  for delta in $(seq 1.01 0.01 1.2)
  do
    M=100000
    X="nY=$nY,delta=$delta,M=$M"
    echo $X
    sbatch -J "$X" --export="$X" submit_rydberg
  done  
  sleep 0.5s
done

