#!/bin/bash 
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRAR_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

module load cuda/12.1
module load gcc/11.2

python /HOME/scw6493/run/code/train_reg_18.py 

