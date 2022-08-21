# #!/bin/bash


# FSS Dataset

joblist=$(sq -h --format="%j")
for nX in $(seq 8 4 20)
do
    for delta in $(seq 1.0 0.01 1.2) $(seq 1.22 0.02 1.28)
    do
        for seed in $(seq 1000 1000 10000)
        do
            X="$nX|$delta|$seed"
            # sbatch -J "$X" --export="nX=$nX,seed=$seed,delta=$delta" submit_rydberg_fss

            if `echo $joblist | grep -owq "$X"` ; then
                # echo "queueing $X"
                # sbatch -J "$X" --export="nX=$nX,seed=$seed,delta=$delta" --dependency=singleton submit_rydberg_fss
        	    # sleep 0.5s
                :
            else
                echo "run $X"
         	    sbatch -J "$X" --export="nX=$nX,seed=$seed,delta=$delta" submit_rydberg_fss
                #:
            fi
        done 
        sleep 0.5s
    done
done

# Kagome
# for nX in $(seq 4 2 8)
# do
#   for M in $(seq 50000 10000 100000)
#   do
#     for seed in $(seq 1000 1000 5000)
#     do
#       for Rb in 1.7 1.95
#       do
#         delta=3.3
#         X="$nX|$Rb|2|$M|$seed"
#         sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2" submit_rydberg

#         X="$nX|$Rb|2+|$M|$seed"
#         sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=2.00000001" submit_rydberg

#         X="$nX|$Rb|4|$M|$seed"
#         sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4" submit_rydberg

#         X="$nX|$Rb|4+|$M|$seed"
#         sbatch -J "$X" --export="M=$M,seed=$seed,nX=$nX,delta=$delta,Rb=$Rb,trunc=4.00000001" submit_rydberg

#         sleep 0.5s
#       done
#     done
#     sleep 0.5s
#   done
# done


# SNN Data

# joblist=$(sq -h --format="%.12j")

# for Rb in $(seq 1.0 0.1 2.0)
# do
#   for delta in $(seq 0.0 0.1 2.9)
#   do
#     for seed in 1234 4321 5555
#     do
#       X="$Rb|$delta|$seed"
#       #echo $X
#       #sbatch -J "$X" --export="seed=$seed,delta=$delta,Rb=$Rb" submit_rydberg

#       if `echo $joblist | grep -owq "$X"` ; then
#         echo "queueing $X"
#         sbatch -J "$X" --export="seed=$seed,delta=$delta,Rb=$Rb" --dependency=singleton submit_rydberg
# 	sleep 0.5s
#       else
#         #echo "run $X"
#  	:
#       fi
#     done
#     #sleep 0.5s
#   done
#   #sleep 1s
# done


#for p in 0.0 0.5 1.0
#do
#  for eps in 0.1
#  do
#    for delta in $(seq -3.0 0.1 3.0)
#    do
#        X="SP|$p|$delta"
#        echo $X
#        sbatch -J "$X" --export="delta=$delta,p=$p,eps=$eps" submit_rydberg_scipost
#    done
#    sleep 1s
#  done
#done

# for delta in $(seq -3.0 0.1 3.0)
# do
#     X="R-$delta"
#     echo $X
#     sbatch -J "$X" --export="delta=$delta" submit_rydberg_kagome
# done

#for nX in (4,6,8), trunc in (2.0, 2.00000001, 4.0, 4.00000001), Rb in (1.7, 1.95)

#for nX in 4 6 8
#do
#    for trunc in 2.0 2.00000001 4.0 4.00000001
#    do
#        for Rb in 1.7 1.95
#        do
#            X="Kag|$nX|$Rb|$trunc"
#            echo $X
#            sbatch -J "$X" --export="nX=$nX,Rb=$Rb,trunc=$trunc" submit_rydberg_kagome
#        done
#    done
#done

# for trunc in $(seq 0 20)
# do
#   for p in 0.0 1.0
#   do
#     X="TR|$trunc|$p|.0"
#     echo $X
#     sbatch -J "$X" --export="trunc=$trunc,p=$p,eps=0.0" submit_rydberg

#     X="TR|$trunc|$p|.1"
#     echo $X
#     sbatch -J "$X" --export="trunc=$trunc,p=$p,eps=0.1" submit_rydberg
#   done
#   sleep 1s
#   # sbatch -J "SP-R|$delta" --export="delta=$delta" submit_rydberg_kagome
# done
# sbatch -J "TR|20|1.0|.1" --export="trunc=20,p=1.0,eps=0.1" submit_rydberg
# sbatch -J "TR|9|1.0|.0" --export="trunc=9,p=1.0,eps=0.0" submit_rydberg
# sbatch -J "TR|11|0.0|.0" --export="trunc=11,p=0.0,eps=0.0" submit_rydberg
# sbatch -J "TR-R|1.1" --export="delta=1.1" submit_rydberg_kagome
