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

python3 -m benchmark.run_all_benchmarks --steps 1000 --dtype float32 --trials 1000 --parallelism 256 --seeds 1 --difficulties "medium" --opt="MARSAdamW" --opt="AdamW" --opt "mars-AdamW" --opt="STORM" --timeout 155000 --exclude "beale.py,rosenbrock.py,rastrigin.py,quadratic_varying_scale.py,quadratic_varying_target.py,saddle_point.py,discontinuous_gradient.py,wide_linear.py,minimax.py,plateau_navigation.py,scale_invariant.py,momentum_utilization.py,batch_size_scaling.py,sparse_gradient.py,layer_wise_scale.py,parameter_scale.py,gradient_delay.py,gradient_noise_scale.py,adversarial_gradient.py,dynamic_landscape.py,constrained_optimization.py"
