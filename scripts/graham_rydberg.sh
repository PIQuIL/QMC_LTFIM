#!/bin/bash
for delta in $(seq 1.02 0.01 1.2)
do
  X="delta=$delta"
  #echo $X
  sbatch -J "Ryd_D2C_contd" --export="$X" continue_rydberg
  sleep 1s
done

