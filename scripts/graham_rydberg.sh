#!/bin/bash
for delta in $(seq 1.0 0.01 1.3)
do
  X="omega=1.0,Rb=1.2,delta=${delta}"
  sbatch -J "$X" --export="$X" submit_rydberg
done

