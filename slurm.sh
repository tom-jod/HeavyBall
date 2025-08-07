#!/bin/bash

#SBATCH --job-name="tom_heavyball"
#SBATCH --gres=shard:6 # 6 shards = 1 GPU
#SBATCH --cpus-per-task=2 # ML1 has ? CPUs
#SBATCH --mem=10GB # ML2 has RealMemory=512000 (MB)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tom.jodrell@physicsx.ai
#SBATCH -o /home/tomjodrell/HeavyBall/logs/100_slurm-%j.out


PROJECT_DIR=HeavyBall
CONDA_ENV_NAME=heavyball
export CUDA_VISIBLE_DEVICES=2
cd $PROJECT_DIR
echo Current directory $(pwd)
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
source /home/tomjodrell/.bashrc
conda activate $CONDA_ENV_NAME
taskset -c 24-31 python benchmark/benchmark_runner.py fastMRI.py "ExternalDistributedShampoo,NAdamW" --runs-per-optimizer=1 --runtime-limit=10 --trials=1 --step-hint=67000
