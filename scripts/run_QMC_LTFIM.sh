#!/bin/bash

for N in 6 8
do 
    for X in 1
    do 
        for Z in 0 1 2 3
        do 
            for b in $(seq 0.5 0.5 5.0)
            do julia main.jl mixedstate $N --zfield $Z --xfield $X --beta $b -M 10000 -s 10 
                
            done
        done
    done
done
