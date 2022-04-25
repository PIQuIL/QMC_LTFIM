# #!/bin/bash

# for nX in $(seq 4 2 8)
# do
#   # for M in $(seq 10000 5000 40000)
#   # do
#     M=40000
#     for seed in $(seq 1000 1000 5000)
#     do
#       Rb=1.7
#       delta=3.3
#       # X="$M|$Rb|2|$seed"
#       # sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2" submit_rydberg

#       # X="$M|$Rb|2+|$seed"
#       # sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2.00000001" submit_rydberg

#       X="$M|$Rb|4|$seed"
#       sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4" submit_rydberg

#       X="$M|$Rb|4+|$seed"
#       sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4.00000001" submit_rydberg

#       sleep 0.5s

#       Rb=1.95
#       # X="$M|$Rb|2|$seed"
#       # sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2" submit_rydberg

#       # X="$M|$Rb|2+|$seed"
#       # sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2.00000001" submit_rydberg

#       X="$M|$Rb|4|$seed"
#       sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4" submit_rydberg

#       X="$M|$Rb|4+|$seed"
#       sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4.00000001" submit_rydberg

#       sleep 0.5s
#     done
#     sleep 0.5s
#   # done
# done

# for Rb in $(seq 1.0 0.1 2.0)
# do
#   for delta in $(seq 0.0 0.1 0.4)
#   do
#     for seed in 1234 4321 5555
#     do
#       X="$Rb|$delta|$seed"
#       echo $X
#       sbatch -J "$X" --export="seed=$seed,delta=$delta,Rb=$Rb" submit_rydberg
#     done
#     sleep 0.5s
#   done
#   for delta in $(seq 2.6 0.1 5.0)
#   do
#     for seed in 1234 4321 5555
#     do
#       X="$Rb|$delta|$seed"
#       echo $X
#       sbatch -J "$X" --export="seed=$seed,delta=$delta,Rb=$Rb" submit_rydberg
#     done
#     sleep 0.5s
#   done
# done

for p in 0.0 1.0
do
  for eps in 0.0 0.01 0.1
  do
    X="$delta/$p/$eps"
    echo $X
    sbatch -J "$X" --export="delta=1.1,p=$p,eps=$eps" submit_rydberg_runstats
  done
done
