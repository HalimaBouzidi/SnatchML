#!/bin/bash

models=(simple resnet mobilenet transformer)

seeds=(1)
expansions=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 1.0) 


for model in ${models[@]}
do 
	for seed in ${seeds[@]}
	do
		idx=0
		for expand in ${expansions[@]}
		do
		  echo "Test for the model: "$model" | expand: "$expand" | seed: "$seed" | idx: "$idx""
		  python3 hijack_er.py --seed $seed --setting black --model $model --expand $expand --hijack-dataset $1 --idx $idx ; 
		  python3 hijack_er.py --seed $seed --setting white --model $model --expand $expand --hijack-dataset $1 --idx $idx ;
		  idx=$((idx + 1))  
	    done
	done
done

exit 0
