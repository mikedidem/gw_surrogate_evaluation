#!/bin/bash

# ####### LR-LHS 50 #####

# # # ## Stage 1
./train.py  --stage 1 --cuda_index 0 --temporal_strategy 'LHS' --nt 100   
./test.py --stage 1 --cuda_index 0 --temporal_strategy 'LHS' --nt 100


# # ## Stage 2
./train.py  --stage 2 --cuda_index 0 --temporal_strategy 'LHS' --nt 200 --temporal_strategy_prev 'LHS' --nt_prev 100 
./test.py --stage 2 --cuda_index 0 --temporal_strategy 'LHS' --nt 200 --temporal_strategy_prev 'LHS' --nt_prev 100

