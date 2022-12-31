#!/bin/bash
export PYTHONPATH="../":$PYTHONPATH
module load anaconda/2020.11
module load openmpi/4.1.1
module load cuda/10.1
module load nccl/2.9.6-1_cuda10.1
module load cudnn/7.6.5.32_cuda10.1
source activate mindspore
mpirun -n $1 python diffusion_equation.py --g_epochs 6000 --f_epochs 6000 --g_lr 0.01 --f_lr 0.01 --parallel_mode DATA_PARALLEL
unset PYTHONPATH