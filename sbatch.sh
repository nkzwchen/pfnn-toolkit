rm slurm*
sbatch --gpus=$1 run_diffusion_equation.sh $1