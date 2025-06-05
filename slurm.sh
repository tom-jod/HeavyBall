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
python3 -m benchmark.run_all_benchmarks --steps 100000 --dtype float32 --trials 15 --parallelism 256 --seeds 1 --difficulties "trivial" --opt="mars-SFAdamW" --opt="mars-SFAdamWEMA" --opt="SFAdamW" --opt="AdamW" --timeout 155000 --exclude "minimax.py,adversarial_gradient.py,wide_linear.py,xor_sequence.py,xor_digit.py,xor_spot.py,xor_sequence_rnn.py,xor_digit_rnn.py,xor_spot_rnn.py,saddle_point.py,discontinuous_gradient.py,wide_linear.py,minimax.py,plateau_navigation.py,scale_invariant.py,momentum_utilization.py,batch_size_scaling.py,gradient_delay.py,gradient_noise_scale.py,adversarial_gradient.py,dynamic_landscape.py,constrained_optimization.py"