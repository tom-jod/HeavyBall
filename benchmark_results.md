# Benchmark Results
Generated: 2025-06-22 09:59:51.424968
Last updated: 2025-06-22 09:59:51.424982

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamWEMAScheduled | No | Yes | 16/21 | 45026.03s | 24.3 |
| AdamWEMA | No | Yes | 17/21 | 46521.79s | 22.1 |
| AdamW | No | Yes | 17/21 | 46317.58s | 22.1 |
| AdamWScheduled | No | Yes | 16/21 | 44289.05s | 24.3 |
| AdamW | No | No | 16/21 | 36240.60s | 18.3 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| batch_size_scaling-medium | AdamW | No | No | ✓ | 2133.62s | 1.85e-16 | 6 | 0 | `ForeachAdamW(lr=0.01780, betas=(0.993, 1.0000), shampoo_beta=0.952)` |
| batch_size_scaling-medium | AdamW | No | Yes | ✓ | 1560.63s | 1.34e-16 | 6 | 0 | `ForeachAdamW(lr=0.02065, betas=(0.982, 1.0000), shampoo_beta=0.359)` |
| batch_size_scaling-medium | AdamWEMA | No | Yes | ✓ | 1557.15s | 1.34e-16 | 6 | 0 | `ForeachAdamWEMA(lr=0.02065, betas=(0.982, 1.0000), shampoo_beta=0.359)` |
| batch_size_scaling-medium | AdamWEMAScheduled | No | Yes | ✓ | 1570.07s | 1.20e-16 | 7 | 0 | `ForeachAdamWEMAScheduled(lr=0.01682, betas=(0.976, 1.0000), shampoo_beta=0.950)` |
| batch_size_scaling-medium | AdamWScheduled | No | Yes | ✓ | 1568.16s | 1.20e-16 | 7 | 0 | `ForeachAdamWScheduled(lr=0.01682, betas=(0.976, 1.0000), shampoo_beta=0.950)` |
| constrained_optimization-medium | AdamW | No | No | ✓ | 19301.90s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | AdamW | No | Yes | ✓ | 20222.26s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | AdamWEMA | No | Yes | ✓ | 20429.61s | 1.00e+00 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | AdamWEMAScheduled | No | Yes | ✓ | 20830.89s | 1.00e+00 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | AdamWScheduled | No | Yes | ✓ | 17991.50s | 1.00e+00 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| dynamic_landscape-medium | AdamW | No | No | ✓ | 18690.09s | 9.83e-03 | 7 | 0 | `ForeachAdamW(lr=2.85073, betas=(0.886, 0.8235), shampoo_beta=1.000)` |
| dynamic_landscape-medium | AdamW | No | Yes | ✓ | 9479.29s | 9.80e-03 | 12 | 0 | `ForeachAdamW(lr=5.57468, betas=(0.950, 1.0000), shampoo_beta=0.999)` |
| dynamic_landscape-medium | AdamWEMA | No | Yes | ✓ | 9484.50s | 9.80e-03 | 12 | 0 | `ForeachAdamWEMA(lr=5.57468, betas=(0.950, 1.0000), shampoo_beta=0.999)` |
| dynamic_landscape-medium | AdamWEMAScheduled | No | Yes | ✓ | 7313.66s | 9.97e-03 | 6 | 0 | `ForeachAdamWEMAScheduled(lr=3.51189, betas=(0.989, 0.9998), shampoo_beta=0.999)` |
| dynamic_landscape-medium | AdamWScheduled | No | Yes | ✓ | 7330.67s | 9.97e-03 | 6 | 0 | `ForeachAdamWScheduled(lr=3.51189, betas=(0.989, 0.9998), shampoo_beta=0.999)` |
| gradient_delay-medium | AdamW | No | No | ✗ | 0.66s | inf | 200 | 0 | N/A |
| gradient_delay-medium | AdamW | No | Yes | ✗ | 0.50s | inf | 200 | 0 | N/A |
| gradient_delay-medium | AdamWEMA | No | Yes | ✗ | 0.27s | inf | 200 | 0 | N/A |
| gradient_delay-medium | AdamWEMAScheduled | No | Yes | ✗ | 0.99s | inf | 200 | 0 | N/A |
| gradient_delay-medium | AdamWScheduled | No | Yes | ✗ | 0.17s | inf | 200 | 0 | N/A |
| gradient_noise_scale-medium | AdamW | No | No | ✓ | 25042.15s | 1.54e-06 | 10 | 0 | `ForeachAdamW(lr=0.01014, betas=(0.981, 1.0000), shampoo_beta=0.050)` |
| gradient_noise_scale-medium | AdamW | No | Yes | ✓ | 27133.08s | 2.40e-06 | 10 | 0 | `ForeachAdamW(lr=0.03495, betas=(0.922, 1.0000), shampoo_beta=0.309)` |
| gradient_noise_scale-medium | AdamWEMA | No | Yes | ✓ | 26762.37s | 2.40e-06 | 10 | 0 | `ForeachAdamWEMA(lr=0.03495, betas=(0.922, 1.0000), shampoo_beta=0.309)` |
| gradient_noise_scale-medium | AdamWEMAScheduled | No | Yes | ✓ | 42574.71s | 1.00e-06 | 38 | 0 | `ForeachAdamWEMAScheduled(lr=0.00050, betas=(1.000, 1.0000), shampoo_beta=0.677)` |
| gradient_noise_scale-medium | AdamWScheduled | No | Yes | ✓ | 39387.68s | 1.00e-06 | 38 | 0 | `ForeachAdamWScheduled(lr=0.00050, betas=(1.000, 1.0000), shampoo_beta=0.677)` |
| layer_wise_scale-medium | AdamW | No | No | ✓ | 27090.56s | 1.00e-04 | 11 | 0 | `ForeachAdamW(lr=0.00065, betas=(1.000, 1.0000), shampoo_beta=0.675)` |
| layer_wise_scale-medium | AdamW | No | Yes | ✓ | 23677.98s | 9.66e-05 | 13 | 0 | `ForeachAdamW(lr=0.01042, betas=(0.772, 1.0000), shampoo_beta=0.517)` |
| layer_wise_scale-medium | AdamWEMA | No | Yes | ✓ | 23515.57s | 9.66e-05 | 13 | 0 | `ForeachAdamWEMA(lr=0.01042, betas=(0.772, 1.0000), shampoo_beta=0.517)` |
| layer_wise_scale-medium | AdamWEMAScheduled | No | Yes | ✓ | 51664.79s | 9.93e-05 | 14 | 0 | `ForeachAdamWEMAScheduled(lr=0.00578, betas=(0.212, 0.9967), shampoo_beta=0.888)` |
| layer_wise_scale-medium | AdamWScheduled | No | Yes | ✓ | 51943.41s | 9.93e-05 | 14 | 0 | `ForeachAdamWScheduled(lr=0.00578, betas=(0.212, 0.9967), shampoo_beta=0.888)` |
| momentum_utilization-medium | AdamW | No | No | ✓ | 92382.55s | -1.36e-04 | 132 | 0 | `ForeachAdamW(lr=0.00110, betas=(0.000, 0.0000), shampoo_beta=0.000)` |
| momentum_utilization-medium | AdamW | No | Yes | ✓ | 119462.36s | -1.50e-05 | 132 | 0 | `ForeachAdamW(lr=0.00045, betas=(0.000, 0.0000), shampoo_beta=0.000)` |
| momentum_utilization-medium | AdamWEMA | No | Yes | ✓ | 120500.37s | -1.50e-05 | 132 | 0 | `ForeachAdamWEMA(lr=0.00045, betas=(0.000, 0.0000), shampoo_beta=0.000)` |
| momentum_utilization-medium | AdamWEMAScheduled | No | Yes | ✗ | 120608.92s | 4.93e-04 | 200 | 0 | `ForeachAdamWEMAScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-medium | AdamWScheduled | No | Yes | ✗ | 116295.71s | 4.93e-04 | 200 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-medium | AdamW | No | No | ✗ | 157046.43s | inf | 200 | 0 | N/A |
| noisy_matmul-medium | AdamW | No | Yes | ✗ | 156062.42s | inf | 200 | 0 | N/A |
| noisy_matmul-medium | AdamWEMA | No | Yes | ✗ | 155969.15s | inf | 200 | 0 | N/A |
| noisy_matmul-medium | AdamWEMAScheduled | No | Yes | ✗ | 155735.77s | inf | 200 | 0 | N/A |
| noisy_matmul-medium | AdamWScheduled | No | Yes | ✗ | 155481.20s | inf | 200 | 0 | N/A |
| parameter_scale-medium | AdamW | No | No | ✓ | 68465.52s | 9.99e-05 | 26 | 0 | `ForeachAdamW(lr=1.21603, betas=(0.961, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-medium | AdamW | No | Yes | ✓ | 83129.04s | 9.59e-05 | 16 | 0 | `ForeachAdamW(lr=3.43573, betas=(0.994, 0.9687), shampoo_beta=1.000)` |
| parameter_scale-medium | AdamWEMA | No | Yes | ✓ | 83225.20s | 9.59e-05 | 16 | 0 | `ForeachAdamWEMA(lr=3.43573, betas=(0.994, 0.9687), shampoo_beta=1.000)` |
| parameter_scale-medium | AdamWEMAScheduled | No | Yes | ✓ | 111735.60s | 9.68e-05 | 61 | 0 | `ForeachAdamWEMAScheduled(lr=1.63032, betas=(0.972, 0.9955), shampoo_beta=1.000)` |
| parameter_scale-medium | AdamWScheduled | No | Yes | ✓ | 108363.69s | 9.68e-05 | 61 | 0 | `ForeachAdamWScheduled(lr=1.63032, betas=(0.972, 0.9955), shampoo_beta=1.000)` |
| plateau_navigation-medium | AdamW | No | No | ✓ | 26399.57s | 7.90e-07 | 3 | 0 | `ForeachAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| plateau_navigation-medium | AdamW | No | Yes | ✓ | 26775.02s | 7.93e-07 | 3 | 0 | `ForeachAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| plateau_navigation-medium | AdamWEMA | No | Yes | ✓ | 27122.42s | 7.93e-07 | 3 | 0 | `ForeachAdamWEMA(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| plateau_navigation-medium | AdamWEMAScheduled | No | Yes | ✓ | 27279.45s | 7.84e-07 | 3 | 0 | `ForeachAdamWEMAScheduled(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| plateau_navigation-medium | AdamWScheduled | No | Yes | ✓ | 26868.02s | 7.84e-07 | 3 | 0 | `ForeachAdamWScheduled(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| quadratic_varying_scale-medium | AdamW | No | No | ✓ | 41826.36s | 1.22e-14 | 24 | 0 | `ForeachAdamW(lr=0.00174, betas=(0.998, 1.0000), shampoo_beta=0.296)` |
| quadratic_varying_scale-medium | AdamW | No | Yes | ✓ | 80700.33s | 1.51e-14 | 26 | 0 | `ForeachAdamW(lr=0.00295, betas=(0.952, 1.0000), shampoo_beta=0.996)` |
| quadratic_varying_scale-medium | AdamWEMA | No | Yes | ✓ | 80820.40s | 1.51e-14 | 26 | 0 | `ForeachAdamWEMA(lr=0.00295, betas=(0.952, 1.0000), shampoo_beta=0.996)` |
| quadratic_varying_scale-medium | AdamWEMAScheduled | No | Yes | ✓ | 33600.48s | 1.53e-14 | 15 | 0 | `ForeachAdamWEMAScheduled(lr=0.00160, betas=(0.994, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_scale-medium | AdamWScheduled | No | Yes | ✓ | 34026.81s | 1.53e-14 | 15 | 0 | `ForeachAdamWScheduled(lr=0.00160, betas=(0.994, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_target-medium | AdamW | No | No | ✓ | 43427.52s | 1.00e-16 | 11 | 0 | `ForeachAdamW(lr=0.00750, betas=(0.996, 1.0000), shampoo_beta=0.804)` |
| quadratic_varying_target-medium | AdamW | No | Yes | ✓ | 105692.54s | 1.00e-16 | 24 | 0 | `ForeachAdamW(lr=0.04802, betas=(0.989, 0.9996), shampoo_beta=1.000)` |
| quadratic_varying_target-medium | AdamWEMA | No | Yes | ✓ | 105593.65s | 1.00e-16 | 24 | 0 | `ForeachAdamWEMA(lr=0.04802, betas=(0.989, 0.9996), shampoo_beta=1.000)` |
| quadratic_varying_target-medium | AdamWEMAScheduled | No | Yes | ✓ | 130915.42s | 1.01e-16 | 23 | 0 | `ForeachAdamWEMAScheduled(lr=0.00195, betas=(0.980, 0.9998), shampoo_beta=0.810)` |
| quadratic_varying_target-medium | AdamWScheduled | No | Yes | ✓ | 127683.96s | 1.01e-16 | 23 | 0 | `ForeachAdamWScheduled(lr=0.00195, betas=(0.980, 0.9998), shampoo_beta=0.810)` |
| saddle_point-medium | AdamW | No | No | ✓ | 24857.77s | 9.99e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | AdamW | No | Yes | ✓ | 26015.34s | 9.99e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | AdamWEMA | No | Yes | ✓ | 25917.72s | 9.99e-02 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | AdamWEMAScheduled | No | Yes | ✓ | 25995.74s | 9.99e-02 | 2 | 0 | `ForeachAdamWEMAScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | AdamWScheduled | No | Yes | ✓ | 25452.18s | 9.99e-02 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-medium | AdamW | No | No | ✓ | 58351.08s | 5.17e-04 | 13 | 0 | `ForeachAdamW(lr=1.40429, betas=(0.917, 0.9974), shampoo_beta=0.982)` |
| scale_invariant-medium | AdamW | No | Yes | ✓ | 53860.49s | 7.82e-04 | 9 | 0 | `ForeachAdamW(lr=1.78752, betas=(0.970, 0.9317), shampoo_beta=1.000)` |
| scale_invariant-medium | AdamWEMA | No | Yes | ✓ | 54483.05s | 7.82e-04 | 9 | 0 | `ForeachAdamWEMA(lr=1.78752, betas=(0.970, 0.9317), shampoo_beta=1.000)` |
| scale_invariant-medium | AdamWEMAScheduled | No | Yes | ✓ | 50199.56s | 9.93e-04 | 12 | 0 | `ForeachAdamWEMAScheduled(lr=2.75657, betas=(0.976, 0.9389), shampoo_beta=0.993)` |
| scale_invariant-medium | AdamWScheduled | No | Yes | ✓ | 49709.48s | 9.93e-04 | 12 | 0 | `ForeachAdamWScheduled(lr=2.75657, betas=(0.976, 0.9389), shampoo_beta=0.993)` |
| sparse_gradient-medium | AdamW | No | No | ✓ | 2326.04s | 9.91e-05 | 7 | 0 | `ForeachAdamW(lr=0.00445, betas=(0.999, 1.0000), shampoo_beta=1.000)` |
| sparse_gradient-medium | AdamW | No | Yes | ✓ | 4072.35s | 9.79e-05 | 6 | 0 | `ForeachAdamW(lr=0.00202, betas=(1.000, 1.0000), shampoo_beta=0.246)` |
| sparse_gradient-medium | AdamWEMA | No | Yes | ✓ | 4137.39s | 9.79e-05 | 6 | 0 | `ForeachAdamWEMA(lr=0.00202, betas=(1.000, 1.0000), shampoo_beta=0.246)` |
| sparse_gradient-medium | AdamWEMAScheduled | No | Yes | ✓ | 2218.53s | 9.63e-05 | 6 | 0 | `ForeachAdamWEMAScheduled(lr=0.00295, betas=(0.915, 1.0000), shampoo_beta=0.210)` |
| sparse_gradient-medium | AdamWScheduled | No | Yes | ✓ | 2196.58s | 9.63e-05 | 6 | 0 | `ForeachAdamWScheduled(lr=0.00295, betas=(0.915, 1.0000), shampoo_beta=0.210)` |
| xor_digit-medium | AdamW | No | No | ✗ | 160512.28s | inf | 200 | 0 | N/A |
| xor_digit-medium | AdamW | No | Yes | ✓ | 120492.53s | 9.86e-04 | 74 | 0 | `ForeachAdamW(lr=0.00712, betas=(0.458, 0.9735), shampoo_beta=0.009)` |
| xor_digit-medium | AdamWEMA | No | Yes | ✓ | 120700.08s | 9.86e-04 | 74 | 0 | `ForeachAdamWEMA(lr=0.00712, betas=(0.458, 0.9735), shampoo_beta=0.009)` |
| xor_digit-medium | AdamWEMAScheduled | No | Yes | ✓ | 121799.60s | 6.08e-04 | 132 | 0 | `ForeachAdamWEMAScheduled(lr=0.00680, betas=(0.000, 0.0000), shampoo_beta=0.000)` |
| xor_digit-medium | AdamWScheduled | No | Yes | ✓ | 118922.46s | 6.08e-04 | 132 | 0 | `ForeachAdamWScheduled(lr=0.00680, betas=(0.000, 0.0000), shampoo_beta=0.000)` |
| xor_digit_rnn-medium | AdamW | No | No | ✗ | 139755.87s | 6.86e-01 | 200 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | AdamW | No | Yes | ✗ | 159093.46s | inf | 200 | 0 | N/A |
| xor_digit_rnn-medium | AdamWEMA | No | Yes | ✗ | 158272.40s | inf | 200 | 0 | N/A |
| xor_digit_rnn-medium | AdamWEMAScheduled | No | Yes | ✗ | 158477.98s | inf | 200 | 0 | N/A |
| xor_digit_rnn-medium | AdamWScheduled | No | Yes | ✗ | 159090.66s | inf | 200 | 0 | N/A |
| xor_sequence-medium | AdamW | No | No | ✓ | 19878.79s | 9.66e-03 | 6 | 0 | `ForeachAdamW(lr=0.00127, betas=(1.000, 1.0000), shampoo_beta=0.953)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 37136.64s | 9.22e-03 | 22 | 0 | `ForeachAdamW(lr=0.00419, betas=(1.000, 1.0000), shampoo_beta=0.264)` |
| xor_sequence-medium | AdamWEMA | No | Yes | ✓ | 36910.33s | 9.22e-03 | 22 | 0 | `ForeachAdamWEMA(lr=0.00419, betas=(1.000, 1.0000), shampoo_beta=0.264)` |
| xor_sequence-medium | AdamWEMAScheduled | No | Yes | ✓ | 21967.48s | 8.37e-03 | 12 | 0 | `ForeachAdamWEMAScheduled(lr=0.10086, betas=(0.972, 1.0000), shampoo_beta=0.948)` |
| xor_sequence-medium | AdamWScheduled | No | Yes | ✓ | 22043.45s | 8.37e-03 | 12 | 0 | `ForeachAdamWScheduled(lr=0.10086, betas=(0.972, 1.0000), shampoo_beta=0.948)` |
| xor_sequence_rnn-medium | AdamW | No | No | ✓ | 91237.12s | 9.66e-03 | 11 | 0 | `ForeachAdamW(lr=0.00005, betas=(1.000, 1.0000), shampoo_beta=0.914)` |
| xor_sequence_rnn-medium | AdamW | No | Yes | ✓ | 32284.78s | 9.96e-03 | 6 | 0 | `ForeachAdamW(lr=0.00002, betas=(0.999, 1.0000), shampoo_beta=0.876)` |
| xor_sequence_rnn-medium | AdamWEMA | No | Yes | ✓ | 32192.72s | 9.96e-03 | 6 | 0 | `ForeachAdamWEMA(lr=0.00002, betas=(0.999, 1.0000), shampoo_beta=0.876)` |
| xor_sequence_rnn-medium | AdamWEMAScheduled | No | Yes | ✓ | 36174.85s | 9.93e-03 | 7 | 0 | `ForeachAdamWEMAScheduled(lr=0.00001, betas=(0.933, 0.9975), shampoo_beta=0.548)` |
| xor_sequence_rnn-medium | AdamWScheduled | No | Yes | ✓ | 38803.31s | 9.93e-03 | 7 | 0 | `ForeachAdamWScheduled(lr=0.00001, betas=(0.933, 0.9975), shampoo_beta=0.548)` |
| xor_spot-medium | AdamW | No | No | ✓ | 18438.93s | 6.33e-03 | 22 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.788, 1.0000), shampoo_beta=0.267)` |
| xor_spot-medium | AdamW | No | Yes | ✓ | 15704.14s | 8.83e-03 | 12 | 0 | `ForeachAdamW(lr=0.00092, betas=(0.745, 1.0000), shampoo_beta=0.001)` |
| xor_spot-medium | AdamWEMA | No | Yes | ✓ | 17517.92s | 8.83e-03 | 12 | 0 | `ForeachAdamWEMA(lr=0.00092, betas=(0.745, 1.0000), shampoo_beta=0.001)` |
| xor_spot-medium | AdamWEMAScheduled | No | Yes | ✓ | 34575.70s | 8.68e-03 | 49 | 0 | `ForeachAdamWEMAScheduled(lr=0.01734, betas=(0.972, 0.9999), shampoo_beta=0.772)` |
| xor_spot-medium | AdamWScheduled | No | Yes | ✓ | 36333.53s | 8.68e-03 | 49 | 0 | `ForeachAdamWScheduled(lr=0.01734, betas=(0.972, 0.9999), shampoo_beta=0.772)` |
| xor_spot_rnn-medium | AdamW | No | No | ✗ | 158605.19s | inf | 200 | 0 | N/A |
| xor_spot_rnn-medium | AdamW | No | Yes | ✗ | 158558.46s | inf | 200 | 0 | N/A |
| xor_spot_rnn-medium | AdamWEMA | No | Yes | ✗ | 160147.04s | inf | 200 | 0 | N/A |
| xor_spot_rnn-medium | AdamWEMAScheduled | No | Yes | ✗ | 158536.08s | inf | 200 | 0 | N/A |
| xor_spot_rnn-medium | AdamWScheduled | No | Yes | ✗ | 158047.28s | inf | 200 | 0 | N/A |

## Errors


### gradient_delay-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - mars-AdamWEMA
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - mars-AdamWEMAScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### noisy_matmul-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - mars-AdamWEMA
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - mars-AdamWEMAScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit.py", line 68, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit_rnn-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit_rnn.py", line 69, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit_rnn-medium - mars-AdamWEMA
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit_rnn.py", line 69, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit_rnn-medium - mars-AdamWEMAScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit_rnn.py", line 69, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit_rnn-medium - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit_rnn.py", line 69, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - mars-AdamWEMA
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - mars-AdamWEMAScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```
