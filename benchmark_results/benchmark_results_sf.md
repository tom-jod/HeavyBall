# Benchmark Results
Generated: 2025-06-03 23:13:37.093106
Last updated: 2025-06-03 23:13:37.093112

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SFAdamW | No | Yes | 22/25 | 17063.97s | 4.7 |
| SFAdamWEMA | No | Yes | 22/25 | 17019.45s | 4.7 |
| SFAdamW | No | No | 21/25 | 16261.20s | 5.4 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| batch_size_scaling-trivial | SFAdamW | No | No | ✓ | 5729.91s | 1.29e-16 | 15 | 0 | `ForeachSFAdamW(lr=0.08840, betas=(1.000, 0.9972), shampoo_beta=0.053)` |
| batch_size_scaling-trivial | SFAdamW | No | Yes | ✓ | 1987.26s | 1.07e-16 | 8 | 0 | `ForeachSFAdamW(lr=0.18794, betas=(0.930, 1.0000), shampoo_beta=0.782)` |
| batch_size_scaling-trivial | SFAdamWEMA | No | Yes | ✓ | 1995.53s | 1.07e-16 | 8 | 0 | `ForeachSFAdamWEMA(lr=0.18794, betas=(0.930, 1.0000), shampoo_beta=0.782)` |
| beale-trivial | SFAdamW | No | No | ✓ | 26589.31s | 9.21e-09 | 13 | 0 | `ForeachSFAdamW(lr=64.21445, betas=(0.006, 0.9975), shampoo_beta=0.830)` |
| beale-trivial | SFAdamW | No | Yes | ✓ | 27017.78s | 6.73e-09 | 7 | 0 | `ForeachSFAdamW(lr=99.04932, betas=(0.989, 0.9604), shampoo_beta=0.757)` |
| beale-trivial | SFAdamWEMA | No | Yes | ✓ | 23119.83s | 6.73e-09 | 7 | 0 | `ForeachSFAdamWEMA(lr=99.04932, betas=(0.989, 0.9604), shampoo_beta=0.757)` |
| constrained_optimization-trivial | SFAdamW | No | No | ✓ | 18042.35s | 1.00e+00 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | SFAdamW | No | Yes | ✓ | 16226.58s | 1.00e+00 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | SFAdamWEMA | No | Yes | ✓ | 16234.70s | 1.00e+00 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | SFAdamW | No | No | ✗ | 32073.50s | 1.48e-05 | 50 | 0 | `ForeachSFAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | SFAdamW | No | Yes | ✗ | 31533.06s | 3.41e-06 | 50 | 0 | `ForeachSFAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | SFAdamWEMA | No | Yes | ✗ | 31505.48s | 3.41e-06 | 50 | 0 | `ForeachSFAdamWEMA(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | SFAdamW | No | No | ✓ | 634.48s | 7.15e-03 | 3 | 0 | `ForeachSFAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| dynamic_landscape-trivial | SFAdamW | No | Yes | ✓ | 661.28s | 7.13e-03 | 3 | 0 | `ForeachSFAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| dynamic_landscape-trivial | SFAdamWEMA | No | Yes | ✓ | 658.61s | 7.13e-03 | 3 | 0 | `ForeachSFAdamWEMA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| gradient_delay-trivial | SFAdamW | No | No | ✓ | 20254.23s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_delay-trivial | SFAdamW | No | Yes | ✓ | 21160.07s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_delay-trivial | SFAdamWEMA | No | Yes | ✓ | 18480.70s | 1.00e-04 | 4 | 0 | `ForeachSFAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_noise_scale-trivial | SFAdamW | No | No | ✓ | 24046.99s | 1.95e-06 | 7 | 0 | `ForeachSFAdamW(lr=0.03012, betas=(0.674, 0.9999), shampoo_beta=0.993)` |
| gradient_noise_scale-trivial | SFAdamW | No | Yes | ✓ | 28924.35s | 1.04e-06 | 6 | 0 | `ForeachSFAdamW(lr=0.01673, betas=(0.956, 1.0000), shampoo_beta=0.951)` |
| gradient_noise_scale-trivial | SFAdamWEMA | No | Yes | ✓ | 28913.05s | 1.04e-06 | 6 | 0 | `ForeachSFAdamWEMA(lr=0.01673, betas=(0.956, 1.0000), shampoo_beta=0.951)` |
| layer_wise_scale-trivial | SFAdamW | No | No | ✓ | 27614.61s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| layer_wise_scale-trivial | SFAdamW | No | Yes | ✓ | 28667.06s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| layer_wise_scale-trivial | SFAdamWEMA | No | Yes | ✓ | 24154.44s | 1.00e-04 | 4 | 0 | `ForeachSFAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| momentum_utilization-trivial | SFAdamW | No | No | ✓ | 27497.65s | -6.38e-06 | 14 | 0 | `ForeachSFAdamW(lr=0.04849, betas=(0.781, 0.9991), shampoo_beta=0.999)` |
| momentum_utilization-trivial | SFAdamW | No | Yes | ✓ | 27292.04s | -1.30e-06 | 8 | 0 | `ForeachSFAdamW(lr=0.05959, betas=(0.986, 1.0000), shampoo_beta=0.996)` |
| momentum_utilization-trivial | SFAdamWEMA | No | Yes | ✓ | 27346.45s | -1.30e-06 | 8 | 0 | `ForeachSFAdamWEMA(lr=0.05959, betas=(0.986, 1.0000), shampoo_beta=0.996)` |
| noisy_matmul-trivial | SFAdamW | No | No | ✗ | 23325.75s | 6.78e-14 | 50 | 0 | `ForeachSFAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | SFAdamW | No | Yes | ✓ | 16130.28s | 1.24e-14 | 11 | 0 | `ForeachSFAdamW(lr=10.27253, betas=(0.997, 1.0000), shampoo_beta=0.729)` |
| noisy_matmul-trivial | SFAdamWEMA | No | Yes | ✓ | 18276.90s | 1.24e-14 | 11 | 0 | `ForeachSFAdamWEMA(lr=10.27253, betas=(0.997, 1.0000), shampoo_beta=0.729)` |
| parameter_scale-trivial | SFAdamW | No | No | ✗ | 41267.78s | 1.75e-02 | 50 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | SFAdamW | No | Yes | ✗ | 41260.68s | 2.06e-02 | 50 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | SFAdamWEMA | No | Yes | ✗ | 40874.37s | 2.06e-02 | 50 | 0 | `ForeachSFAdamWEMA(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| plateau_navigation-trivial | SFAdamW | No | No | ✓ | 15848.46s | 9.99e-05 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | SFAdamW | No | Yes | ✓ | 16501.96s | 9.99e-05 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | SFAdamWEMA | No | Yes | ✓ | 16518.58s | 9.99e-05 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | SFAdamW | No | No | ✓ | 23046.32s | 6.01e-14 | 10 | 0 | `ForeachSFAdamW(lr=0.89992, betas=(0.985, 0.9999), shampoo_beta=0.971)` |
| quadratic_varying_scale-trivial | SFAdamW | No | Yes | ✓ | 23430.99s | 1.18e-14 | 10 | 0 | `ForeachSFAdamW(lr=0.26674, betas=(0.470, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | SFAdamWEMA | No | Yes | ✓ | 27548.42s | 1.18e-14 | 10 | 0 | `ForeachSFAdamWEMA(lr=0.26674, betas=(0.470, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | SFAdamW | No | No | ✓ | 23398.49s | 5.33e-15 | 9 | 0 | `ForeachSFAdamW(lr=76.32257, betas=(1.000, 1.0000), shampoo_beta=0.211)` |
| quadratic_varying_target-trivial | SFAdamW | No | Yes | ✓ | 23518.88s | 7.11e-15 | 8 | 0 | `ForeachSFAdamW(lr=61.71814, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| quadratic_varying_target-trivial | SFAdamWEMA | No | Yes | ✓ | 27732.51s | 7.11e-15 | 8 | 0 | `ForeachSFAdamWEMA(lr=61.71814, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| rastrigin-trivial | SFAdamW | No | No | ✗ | 35146.95s | 3.98e+00 | 50 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SFAdamW | No | Yes | ✗ | 34739.64s | 3.98e+00 | 50 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SFAdamWEMA | No | Yes | ✗ | 35908.62s | 3.98e+00 | 50 | 0 | `ForeachSFAdamWEMA(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | SFAdamW | No | No | ✓ | 21872.88s | 6.08e-10 | 6 | 0 | `ForeachSFAdamW(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| rosenbrock-trivial | SFAdamW | No | Yes | ✓ | 22631.33s | 2.22e-10 | 6 | 0 | `ForeachSFAdamW(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| rosenbrock-trivial | SFAdamWEMA | No | Yes | ✓ | 26593.65s | 2.22e-10 | 6 | 0 | `ForeachSFAdamWEMA(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| saddle_point-trivial | SFAdamW | No | No | ✓ | 18562.05s | 9.99e-02 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | SFAdamW | No | Yes | ✓ | 17007.28s | 9.99e-02 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | SFAdamWEMA | No | Yes | ✓ | 17030.93s | 9.99e-02 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | SFAdamW | No | No | ✓ | 25285.68s | 9.98e-04 | 6 | 0 | `ForeachSFAdamW(lr=6.61125, betas=(0.993, 0.8098), shampoo_beta=1.000)` |
| scale_invariant-trivial | SFAdamW | No | Yes | ✓ | 26220.79s | 9.99e-04 | 6 | 0 | `ForeachSFAdamW(lr=6.61125, betas=(0.993, 0.8098), shampoo_beta=1.000)` |
| scale_invariant-trivial | SFAdamWEMA | No | Yes | ✓ | 26248.49s | 9.99e-04 | 6 | 0 | `ForeachSFAdamWEMA(lr=6.61125, betas=(0.993, 0.8098), shampoo_beta=1.000)` |
| sparse_gradient-trivial | SFAdamW | No | No | ✓ | 3822.32s | 9.99e-05 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | SFAdamW | No | Yes | ✓ | 3978.63s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | SFAdamWEMA | No | Yes | ✓ | 4114.42s | 1.00e-04 | 4 | 0 | `ForeachSFAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_digit-trivial | SFAdamW | No | No | ✓ | 2568.35s | 9.93e-04 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | SFAdamW | No | Yes | ✓ | 3057.15s | 9.98e-04 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | SFAdamWEMA | No | Yes | ✓ | 3065.64s | 9.98e-04 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | SFAdamW | No | No | ✓ | 6553.74s | 9.34e-04 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | SFAdamW | No | Yes | ✓ | 7724.73s | 9.38e-04 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | SFAdamWEMA | No | Yes | ✓ | 7827.34s | 9.38e-04 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | SFAdamW | No | No | ✓ | 19438.49s | 9.68e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | SFAdamW | No | Yes | ✓ | 22748.76s | 9.69e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | SFAdamWEMA | No | Yes | ✓ | 22742.19s | 9.69e-03 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | SFAdamW | No | No | ✓ | 19826.95s | 9.93e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | SFAdamW | No | Yes | ✓ | 27059.02s | 9.94e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | SFAdamWEMA | No | Yes | ✓ | 23178.57s | 9.94e-03 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | SFAdamW | No | No | ✓ | 2040.83s | 9.87e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | SFAdamW | No | Yes | ✓ | 2410.41s | 9.92e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | SFAdamWEMA | No | Yes | ✓ | 2389.58s | 9.92e-03 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | SFAdamW | No | No | ✓ | 8811.06s | 9.52e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | SFAdamW | No | Yes | ✓ | 11050.59s | 9.57e-03 | 2 | 0 | `ForeachSFAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | SFAdamWEMA | No | Yes | ✓ | 10257.30s | 9.57e-03 | 2 | 0 | `ForeachSFAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |

## Errors

