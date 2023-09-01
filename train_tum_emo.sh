#!/bin/bash

cmd="python train.py --model=ctmwa --wnet_lr=0.0005 --inner_lr=0.003 --lr=0.0005 
--trans_ti=0.05 --trans_it=0.05 --trans_tit=0.05 --trans_iti=0.05 --dataset=tum_emo 
--output_dim=7 --niter=10 --niter_decay=10 --batch_size=32
"
echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
eval $cmd
