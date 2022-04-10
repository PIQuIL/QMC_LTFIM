#!/bin/bash

for p in $(seq 0.0 0.5 1.0)
do
  for delta in $(seq -3.0 0.1 3.0)
  do
    eps=0.0
    X="$delta|$p|$eps"
    sbatch -J "$X" --export="delta=$delta,eps=$eps,p=$p" submit_rydberg

    # eps=0.01
    # X="$delta|$p|$eps"
    # sbatch -J "$X" --export="delta=$delta,eps=$eps,p=$p" submit_rydberg

    # eps=0.05
    # X="$delta|$p|$eps"
    # sbatch -J "$X" --export="delta=$delta,eps=$eps,p=$p" submit_rydberg

    # eps=0.1
    # X="$delta|$p|$eps"
    # sbatch -J "$X" --export="delta=$delta,eps=$eps,p=$p" submit_rydberg

    sleep 0.5s
  done
done
