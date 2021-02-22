#!/bin/bash
for N in 4 6 8 10 12
do
  for delta in $(seq 1.0 0.01 1.3)
  do
    X="nY=${N},omega=${1.0},Rb=${1.2},delta=${delta}"
    sbatch -J "$X" --export="$X" --mem=4GB --account=rrg-rgmelko-ab submit
  done
done