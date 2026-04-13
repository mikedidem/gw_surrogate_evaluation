#!/bin/bash

# ## 
./trainer.py  --sigma 25  --constraint 'SOFT' --spatial_strategy 'LR' --filename './well.dat' --temporal_strategy 'LR' --nt 50 --ratio 1.04 --epochs_Adam 1000 --epochs_LBFGS 1000 --lr 0.01 --lam 100 --cuda_index 0


# ## 
./trainer.py  --sigma 25  --constraint 'SOFT' --spatial_strategy 'LR' --filename './well.dat' --temporal_strategy 'LHS' --nt 50  --epochs_Adam 1000 --epochs_LBFGS 1000 --lr 0.01 --lam 100 --cuda_index 0


# ##
./trainer.py  --sigma 25  --constraint 'SOFT' --spatial_strategy 'LR' --filename './well.dat' --temporal_strategy 'UNIFORM' --nt 50  --epochs_Adam 1000 --epochs_LBFGS 1000 --lr 0.01 --lam 100 --cuda_index 0


# ./test.py --sigma 25  --constraint 'SOFT' --spatial_strategy 'LR' --filename './well.dat' --temporal_strategy 'LR' --nt 50 --ratio 1.04
# ./test.py --sigma 25 --constraint 'SOFT' --spatial_strategy 'LR' --filename './well.dat' --temporal_strategy 'LHS' --nt 50
# ./test.py --sigma 25 --constraint 'SOFT' --spatial_strategy 'UNIFORM' --filename './well.dat' --temporal_strategy 'LHS' --nt 50 
