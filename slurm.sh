#!/bin/bash

#SBATCH --job-name="tom_heavyball"
#SBATCH --gres=shard:4 # 6 shards = 1 GPU
#SBATCH --cpus-per-task=2 # ML1 has ? CPUs
#SBATCH --mem=10GB # ML2 has RealMemory=512000 (MB)
#SBATCH --oversubscribe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tom.jodrell@physicsx.ai
#SBATCH -o /home/tomjodrell/HeavyBall/logs/slurm-%j.out


PROJECT_DIR=HeavyBall
CONDA_ENV_NAME=heavyball

cd $PROJECT_DIR
echo Current directory $(pwd)
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
source /home/tomjodrell/.bashrc
conda activate $CONDA_ENV_NAME

python3 -m benchmark.run_all_benchmarks --steps 100000 --dtype float32 --trials 50 --parallelism 256 --seeds 1 --difficulties "trivial" --opt="SGD" --timeout 155000 --exclude "wide_linear.py, minimax.py"