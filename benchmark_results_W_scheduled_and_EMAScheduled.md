# Benchmark Results
Generated: 2025-05-24 04:31:06.188258
Last updated: 2025-05-24 04:31:06.188263

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamWEMAScheduled | No | Yes | 23/28 | 16807.86s | 3.7 |
| AdamW | No | Yes | 21/28 | 17612.45s | 11.4 |
| AdamWScheduled | No | Yes | 21/28 | 15918.21s | 9.5 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | AdamW | No | Yes | ✗ | 5377.86s | -3.83e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| adversarial_gradient-trivial | AdamWEMAScheduled | No | Yes | ✗ | 4915.88s | -4.52e-02 | 50 | 0 | `ForeachAdamWEMAScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| adversarial_gradient-trivial | AdamWScheduled | No | Yes | ✗ | 5137.83s | -4.42e-02 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamW | No | Yes | ✓ | 2700.54s | 1.05e-16 | 6 | 0 | `ForeachAdamW(lr=0.03357, betas=(0.872, 1.0000), shampoo_beta=0.388)` |
| batch_size_scaling-trivial | AdamWEMAScheduled | No | Yes | ✓ | 1401.47s | 1.22e-16 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| batch_size_scaling-trivial | AdamWScheduled | No | Yes | ✓ | 5060.33s | 1.11e-16 | 7 | 0 | `ForeachAdamWScheduled(lr=0.00151, betas=(0.988, 1.0000), shampoo_beta=0.994)` |
| beale-trivial | AdamW | No | Yes | ✗ | 37200.19s | 1.07e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| beale-trivial | AdamWEMAScheduled | No | Yes | ✓ | 32861.83s | 1.00e-08 | 8 | 0 | `ForeachAdamWEMAScheduled(lr=0.20715, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| beale-trivial | AdamWScheduled | No | Yes | ✗ | 28338.51s | 1.28e+01 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-trivial | AdamW | No | Yes | ✓ | 16421.11s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | AdamWEMAScheduled | No | Yes | ✓ | 16577.12s | 1.00e+00 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | AdamWScheduled | No | Yes | ✓ | 15730.19s | 1.00e+00 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | AdamW | No | Yes | ✗ | 26180.20s | 2.74e-05 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | AdamWEMAScheduled | No | Yes | ✓ | 23155.77s | 1.05e-08 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| discontinuous_gradient-trivial | AdamWScheduled | No | Yes | ✗ | 30992.81s | 1.22e-06 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | AdamW | No | Yes | ✓ | 3514.38s | 9.85e-03 | 8 | 0 | `ForeachAdamW(lr=17.56343, betas=(0.985, 1.0000), shampoo_beta=0.981)` |
| dynamic_landscape-trivial | AdamWEMAScheduled | No | Yes | ✓ | 1706.77s | 9.70e-03 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| dynamic_landscape-trivial | AdamWScheduled | No | Yes | ✓ | 1647.80s | 9.87e-03 | 6 | 0 | `ForeachAdamWScheduled(lr=1.89986, betas=(0.946, 0.9997), shampoo_beta=0.903)` |
| gradient_delay-trivial | AdamW | No | Yes | ✓ | 1498.63s | 9.90e-05 | 9 | 0 | `ForeachAdamW(lr=0.01047, betas=(0.960, 1.0000), shampoo_beta=0.959)` |
| gradient_delay-trivial | AdamWEMAScheduled | No | Yes | ✓ | 730.93s | 9.64e-05 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_delay-trivial | AdamWScheduled | No | Yes | ✓ | 2346.85s | 9.95e-05 | 12 | 0 | `ForeachAdamWScheduled(lr=0.00240, betas=(0.982, 1.0000), shampoo_beta=0.396)` |
| gradient_noise_scale-trivial | AdamW | No | Yes | ✓ | 24857.63s | 1.00e-06 | 35 | 0 | `ForeachAdamW(lr=0.00073, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| gradient_noise_scale-trivial | AdamWEMAScheduled | No | Yes | ✓ | 23903.13s | 1.36e-06 | 6 | 0 | `ForeachAdamWEMAScheduled(lr=0.00718, betas=(0.969, 1.0000), shampoo_beta=0.909)` |
| gradient_noise_scale-trivial | AdamWScheduled | No | Yes | ✓ | 20450.21s | 1.14e-06 | 15 | 0 | `ForeachAdamWScheduled(lr=0.00338, betas=(0.993, 1.0000), shampoo_beta=0.480)` |
| layer_wise_scale-trivial | AdamW | No | Yes | ✓ | 28290.68s | 9.89e-05 | 26 | 0 | `ForeachAdamW(lr=0.88257, betas=(0.914, 0.9765), shampoo_beta=0.999)` |
| layer_wise_scale-trivial | AdamWEMAScheduled | No | Yes | ✓ | 23433.87s | 9.92e-05 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| layer_wise_scale-trivial | AdamWScheduled | No | Yes | ✓ | 22997.71s | 1.00e-04 | 19 | 0 | `ForeachAdamWScheduled(lr=0.00067, betas=(1.000, 1.0000), shampoo_beta=0.521)` |
| minimax-trivial | AdamW | No | Yes | ✗ | 17737.08s | 1.56e+00 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-trivial | AdamWEMAScheduled | No | Yes | ✗ | 17455.67s | 2.42e+00 | 50 | 0 | `ForeachAdamWEMAScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-trivial | AdamWScheduled | No | Yes | ✗ | 17674.65s | 1.56e+00 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-trivial | AdamW | No | Yes | ✓ | 19809.88s | -2.83e-05 | 22 | 0 | `ForeachAdamW(lr=0.59148, betas=(0.906, 1.0000), shampoo_beta=0.997)` |
| momentum_utilization-trivial | AdamWEMAScheduled | No | Yes | ✓ | 21362.20s | -1.74e-05 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| momentum_utilization-trivial | AdamWScheduled | No | Yes | ✓ | 21505.53s | -9.17e-05 | 22 | 0 | `ForeachAdamWScheduled(lr=0.73653, betas=(0.845, 0.5996), shampoo_beta=0.999)` |
| noisy_matmul-trivial | AdamW | No | Yes | ✗ | 41282.59s | 1.11e-11 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | AdamWEMAScheduled | No | Yes | ✗ | 32695.60s | 6.42e-14 | 50 | 0 | `ForeachAdamWEMAScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | AdamWScheduled | No | Yes | ✓ | 12213.56s | 1.04e-14 | 15 | 0 | `ForeachAdamWScheduled(lr=0.00182, betas=(0.924, 0.9954), shampoo_beta=0.984)` |
| parameter_scale-trivial | AdamW | No | Yes | ✓ | 36090.50s | 9.59e-05 | 16 | 0 | `ForeachAdamW(lr=3.43573, betas=(0.994, 0.9687), shampoo_beta=1.000)` |
| parameter_scale-trivial | AdamWEMAScheduled | No | Yes | ✗ | 40899.02s | 2.32e-01 | 50 | 0 | `ForeachAdamWEMAScheduled(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | AdamWScheduled | No | Yes | ✗ | 40871.24s | 2.18e-04 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| plateau_navigation-trivial | AdamW | No | Yes | ✓ | 15393.09s | 9.96e-05 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | AdamWEMAScheduled | No | Yes | ✓ | 17750.83s | 1.00e-04 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | AdamWScheduled | No | Yes | ✓ | 15464.96s | 9.96e-05 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | AdamW | No | Yes | ✓ | 40207.43s | 2.14e-14 | 45 | 0 | `ForeachAdamW(lr=0.00171, betas=(0.936, 0.9911), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | AdamWEMAScheduled | No | Yes | ✓ | 22813.45s | 1.22e-14 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| quadratic_varying_scale-trivial | AdamWScheduled | No | Yes | ✓ | 25822.89s | 6.52e-14 | 23 | 0 | `ForeachAdamWScheduled(lr=3.80246, betas=(0.981, 0.9995), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | AdamW | No | Yes | ✓ | 32731.06s | 2.67e-15 | 27 | 0 | `ForeachAdamW(lr=0.98099, betas=(0.967, 0.9986), shampoo_beta=0.103)` |
| quadratic_varying_target-trivial | AdamWEMAScheduled | No | Yes | ✓ | 36882.78s | 8.88e-16 | 10 | 0 | `ForeachAdamWEMAScheduled(lr=0.01250, betas=(0.999, 0.9680), shampoo_beta=0.995)` |
| quadratic_varying_target-trivial | AdamWScheduled | No | Yes | ✓ | 36780.74s | 9.25e-16 | 32 | 0 | `ForeachAdamWScheduled(lr=0.27884, betas=(0.924, 0.9968), shampoo_beta=0.989)` |
| rastrigin-trivial | AdamW | No | Yes | ✗ | 40935.43s | 3.98e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | AdamWEMAScheduled | No | Yes | ✓ | 17692.65s | 1.10e-05 | 3 | 0 | `ForeachAdamWEMAScheduled(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| rastrigin-trivial | AdamWScheduled | No | Yes | ✗ | 36218.62s | 3.98e+00 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | AdamW | No | Yes | ✓ | 22048.31s | 9.96e-10 | 9 | 0 | `ForeachAdamW(lr=0.00465, betas=(0.950, 0.9995), shampoo_beta=0.999)` |
| rosenbrock-trivial | AdamWEMAScheduled | No | Yes | ✓ | 22101.52s | 9.99e-10 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| rosenbrock-trivial | AdamWScheduled | No | Yes | ✓ | 22316.31s | 9.99e-10 | 6 | 0 | `ForeachAdamWScheduled(lr=0.03823, betas=(0.835, 1.0000), shampoo_beta=0.865)` |
| saddle_point-trivial | AdamW | No | Yes | ✓ | 18532.48s | 9.98e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | AdamWEMAScheduled | No | Yes | ✓ | 15733.39s | 9.99e-02 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | AdamWScheduled | No | Yes | ✓ | 18696.35s | 9.98e-02 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | AdamW | No | Yes | ✓ | 21220.48s | 9.94e-04 | 6 | 0 | `ForeachAdamW(lr=0.03245, betas=(0.997, 1.0000), shampoo_beta=0.292)` |
| scale_invariant-trivial | AdamWEMAScheduled | No | Yes | ✓ | 20753.75s | 9.12e-04 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| scale_invariant-trivial | AdamWScheduled | No | Yes | ✓ | 27572.69s | 9.93e-04 | 14 | 0 | `ForeachAdamWScheduled(lr=2.49202, betas=(0.977, 0.9694), shampoo_beta=0.994)` |
| sparse_gradient-trivial | AdamW | No | Yes | ✓ | 766.60s | 9.88e-05 | 8 | 0 | `ForeachAdamW(lr=0.00361, betas=(0.989, 1.0000), shampoo_beta=0.997)` |
| sparse_gradient-trivial | AdamWEMAScheduled | No | Yes | ✓ | 135.86s | 9.93e-05 | 4 | 0 | `ForeachAdamWEMAScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | AdamWScheduled | No | Yes | ✓ | 285.18s | 9.91e-05 | 6 | 0 | `ForeachAdamWScheduled(lr=0.02586, betas=(0.965, 1.0000), shampoo_beta=0.559)` |
| wide_linear-trivial | AdamW | No | Yes | ✗ | 24050.50s | 8.40e-08 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-trivial | AdamWEMAScheduled | No | Yes | ✗ | 20314.79s | 8.38e-08 | 50 | 0 | `ForeachAdamWEMAScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-trivial | AdamWScheduled | No | Yes | ✗ | 27703.20s | 5.47e-08 | 50 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-trivial | AdamW | No | Yes | ✓ | 3506.53s | 9.82e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | AdamWEMAScheduled | No | Yes | ✓ | 2410.51s | 9.59e-04 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | AdamWScheduled | No | Yes | ✓ | 3495.85s | 9.80e-04 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamW | No | Yes | ✓ | 21653.67s | 9.91e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamWEMAScheduled | No | Yes | ✓ | 22638.49s | 9.67e-04 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 21663.58s | 9.88e-04 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamW | No | Yes | ✓ | 25511.49s | 9.34e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamWEMAScheduled | No | Yes | ✓ | 25345.70s | 9.95e-03 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamWScheduled | No | Yes | ✓ | 25497.19s | 9.34e-03 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamW | No | Yes | ✓ | 25607.00s | 9.94e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamWEMAScheduled | No | Yes | ✓ | 25971.50s | 9.98e-03 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 25398.80s | 9.94e-03 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamW | No | Yes | ✓ | 636.55s | 9.29e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot-trivial | AdamWEMAScheduled | No | Yes | ✓ | 1379.80s | 9.86e-03 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamWScheduled | No | Yes | ✓ | 644.29s | 9.76e-03 | 4 | 0 | `ForeachAdamWScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamW | No | Yes | ✓ | 8863.44s | 9.87e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamWEMAScheduled | No | Yes | ✓ | 9837.53s | 9.95e-03 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 8691.45s | 9.65e-03 | 4 | 0 | `ForeachAdamWScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors

