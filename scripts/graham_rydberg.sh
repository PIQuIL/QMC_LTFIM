#!/bin/bash
for delta in $(seq 1.01 0.01 1.2)
do
  X="delta=$delta,nY=$nY"
  sbatch -J "Ryd_Disord2Checkerboard" --export="$X" submit_rydberg
  sleep 0.5s
done
