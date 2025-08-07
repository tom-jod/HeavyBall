python -m benchmark.MNIST --opt AdamW --steps 93750 --trials 10
python -m benchmark.CIFAR10 --opt AdamW --steps 39063 --trials 100
python3 -m benchmark.run_all_benchmarks --steps 93750 --dtype float32 --trials 2 --parallelism 256 --seeds 1 --opt="mars-AdamW" --timeout 15555000 --exclude "beale.py,xor_spot.py,rosenbrock.py,rastrigin.py,quadratic_varying_scale.py,quadratic_varying_target.py,noisy_matmul.py,xor_sequence.py,xor_digit.py,xor_sequence_rnn.py,xor_digit_rnn.py,xor_spot_rnn.py,saddle_point.py,discontinuous_gradient.py,wide_linear.py,minimax.py,plateau_navigation.py,scale_invariant.py,momentum_utilization.py,batch_size_scaling.py,sparse_gradient.py,layer_wise_scale.py,parameter_scale.py,gradient_delay.py,gradient_noise_scale.py,adversarial_gradient.py,dynamic_landscape.py,constrained_optimization.py"
python benchmark/benchmark_runner.py MNIST.py "AdamW" --runs-per-optimizer=5 --steps=1000 --trials=50
python benchmark/benchmark_runner.py MNIST.py "mars-AdamW,AdamW,SGD,SOAP,Muon" --runs-per-optimizer=3 --steps=93750 --trials=20
python benchmark/benchmark_runner.py CIFAR10.py "mars-AdamW,AdamW" --runs-per-optimizer=1 --steps=73750 --trials=5
python benchmark/benchmark_runner.py CIFAR100.py "mars-AdamW_lr_schedule,AdamW_lr_schedule" --runs-per-optimizer=1 --steps=80000 --trials=10
#SBATCH --gres=shard:6 # 6 shards = 1 GPU
python benchmark/benchmark_runner.py Tolstoi_RNN.py "AdamW, mars-AdamW" --runs-per-optimizer=1 --steps=100000 --trials=20

python -m benchmark.car_cfd_transolver --opt COSMOS --steps 1000 --trials 1 2>/dev/null
taskset -c 8-15 python benchmark/benchmark_runner.py MNIST.py "NAdamW" --runs-per-optimizer=1 --steps=5000 --trials=20 