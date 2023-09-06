#!/bin/bash

cmd="python train.py --model=ctmwa --wnet_lr=0.0005 --inner_lr=0.004 --lr=0.0007 --trans_ti=0.5 --trans_it=1.0 
--trans_tit=0.5 --trans_iti=0.05 --dataset=mvsa_mul --niter=4 --niter_decay=4 --batch_size=64"
echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
eval $cmd
