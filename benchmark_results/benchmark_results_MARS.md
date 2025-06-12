# Benchmark Results
Generated: 2025-06-12 01:23:24.765295
Last updated: 2025-06-12 01:23:24.765309

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| STORM_plus | No | No | 4/28 | 11420.84s | 3.2 |
| AdamW | No | No | 21/28 | 18318.60s | 11.4 |
| MARSAdamW | No | No | 22/28 | 22630.78s | 6.0 |
| AdamW | No | Yes | 21/28 | 19500.65s | 11.4 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | AdamW | No | No | ✗ | 6431.38s | -4.58e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| adversarial_gradient-trivial | MARSAdamW | No | No | ✗ | 9220.36s | -1.50e-01 | 50 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| adversarial_gradient-trivial | STORM_plus | No | No | ✓ | 177.39s | -2.29e-04 | 3 | 0 | `ForeachSTORM_plus(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| adversarial_gradient-trivial | AdamW | No | Yes | ✗ | 6561.74s | -3.83e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamW | No | No | ✓ | 3269.43s | 1.08e-16 | 6 | 0 | `ForeachAdamW(lr=0.02090, betas=(0.985, 1.0000), shampoo_beta=0.394)` |
| batch_size_scaling-trivial | MARSAdamW | No | No | ✓ | 7533.63s | 1.17e-16 | 16 | 0 | `ForeachMARSAdamW(lr=0.00277, betas=(0.992, 1.0000), shampoo_beta=0.992)` |
| batch_size_scaling-trivial | STORM_plus | No | No | ✗ | 17318.85s | 1.09e-06 | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamW | No | Yes | ✓ | 3005.47s | 1.05e-16 | 6 | 0 | `ForeachAdamW(lr=0.03357, betas=(0.872, 1.0000), shampoo_beta=0.388)` |
| beale-trivial | AdamW | No | No | ✗ | 28863.44s | 1.31e+01 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| beale-trivial | MARSAdamW | No | No | ✗ | 30231.40s | 9.25e-03 | 50 | 0 | `ForeachMARSAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| beale-trivial | STORM_plus | No | No | ✗ | 7291.23s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| beale-trivial | AdamW | No | Yes | ✗ | 40039.41s | 1.07e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-trivial | AdamW | No | No | ✓ | 16910.30s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | MARSAdamW | No | No | ✓ | 18438.45s | 1.00e+00 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | STORM_plus | No | No | ✗ | 7606.95s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-trivial | AdamW | No | Yes | ✓ | 15566.46s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | AdamW | No | No | ✗ | 37344.73s | 5.17e-07 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | MARSAdamW | No | No | ✗ | 33123.92s | 1.82e-06 | 50 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | STORM_plus | No | No | ✗ | 8344.53s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| discontinuous_gradient-trivial | AdamW | No | Yes | ✗ | 29240.26s | 2.74e-05 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | AdamW | No | No | ✓ | 13513.49s | 9.61e-03 | 6 | 0 | `ForeachAdamW(lr=0.46575, betas=(0.959, 1.0000), shampoo_beta=0.936)` |
| dynamic_landscape-trivial | MARSAdamW | No | No | ✓ | 566.92s | 9.99e-03 | 3 | 0 | `ForeachMARSAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| dynamic_landscape-trivial | STORM_plus | No | No | ✗ | 5634.61s | 4.99e-01 | 50 | 0 | `ForeachSTORM_plus(lr=0.10000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | AdamW | No | Yes | ✓ | 3993.56s | 9.85e-03 | 8 | 0 | `ForeachAdamW(lr=17.56343, betas=(0.985, 1.0000), shampoo_beta=0.981)` |
| gradient_delay-trivial | AdamW | No | No | ✓ | 1324.05s | 9.53e-05 | 7 | 0 | `ForeachAdamW(lr=0.02505, betas=(0.944, 1.0000), shampoo_beta=0.649)` |
| gradient_delay-trivial | MARSAdamW | No | No | ✓ | 29742.42s | 9.98e-05 | 9 | 0 | `ForeachMARSAdamW(lr=0.00343, betas=(0.871, 0.9998), shampoo_beta=0.997)` |
| gradient_delay-trivial | STORM_plus | No | No | ✗ | 8472.35s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| gradient_delay-trivial | AdamW | No | Yes | ✓ | 1719.02s | 9.90e-05 | 9 | 0 | `ForeachAdamW(lr=0.01047, betas=(0.960, 1.0000), shampoo_beta=0.959)` |
| gradient_noise_scale-trivial | AdamW | No | No | ✓ | 25621.62s | 1.00e-06 | 31 | 0 | `ForeachAdamW(lr=0.00066, betas=(0.999, 1.0000), shampoo_beta=0.612)` |
| gradient_noise_scale-trivial | MARSAdamW | No | No | ✓ | 28878.81s | 1.01e-06 | 8 | 0 | `ForeachMARSAdamW(lr=30.08802, betas=(1.000, 0.9992), shampoo_beta=1.000)` |
| gradient_noise_scale-trivial | STORM_plus | No | No | ✓ | 16616.90s | 1.17e-06 | 4 | 0 | `ForeachSTORM_plus(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_noise_scale-trivial | AdamW | No | Yes | ✓ | 27302.60s | 1.00e-06 | 35 | 0 | `ForeachAdamW(lr=0.00073, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| layer_wise_scale-trivial | AdamW | No | No | ✓ | 24580.39s | 1.00e-04 | 22 | 0 | `ForeachAdamW(lr=0.00054, betas=(1.000, 1.0000), shampoo_beta=0.728)` |
| layer_wise_scale-trivial | MARSAdamW | No | No | ✓ | 26525.72s | 9.77e-05 | 8 | 0 | `ForeachMARSAdamW(lr=2.00631, betas=(1.000, 0.0222), shampoo_beta=0.998)` |
| layer_wise_scale-trivial | STORM_plus | No | No | ✗ | 8530.74s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| layer_wise_scale-trivial | AdamW | No | Yes | ✓ | 30426.84s | 9.89e-05 | 26 | 0 | `ForeachAdamW(lr=0.88257, betas=(0.914, 0.9765), shampoo_beta=0.999)` |
| minimax-trivial | AdamW | No | No | ✗ | 21103.34s | 1.58e+00 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-trivial | MARSAdamW | No | No | ✗ | 23745.67s | 1.57e+00 | 50 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-trivial | STORM_plus | No | No | ✗ | 8455.64s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-trivial | AdamW | No | Yes | ✗ | 20292.51s | 1.56e+00 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-trivial | AdamW | No | No | ✓ | 21146.88s | 9.48e-07 | 22 | 0 | `ForeachAdamW(lr=0.00167, betas=(0.997, 1.0000), shampoo_beta=0.857)` |
| momentum_utilization-trivial | MARSAdamW | No | No | ✓ | 24853.13s | -2.86e-06 | 7 | 0 | `ForeachMARSAdamW(lr=5.25508, betas=(0.996, 0.9772), shampoo_beta=1.000)` |
| momentum_utilization-trivial | STORM_plus | No | No | ✓ | 14517.19s | -2.27e-06 | 3 | 0 | `ForeachSTORM_plus(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| momentum_utilization-trivial | AdamW | No | Yes | ✓ | 22447.17s | -2.83e-05 | 22 | 0 | `ForeachAdamW(lr=0.59148, betas=(0.906, 1.0000), shampoo_beta=0.997)` |
| noisy_matmul-trivial | AdamW | No | No | ✓ | 26231.90s | 1.42e-14 | 46 | 0 | `ForeachAdamW(lr=6.51946, betas=(0.993, 1.0000), shampoo_beta=0.671)` |
| noisy_matmul-trivial | MARSAdamW | No | No | ✓ | 15894.59s | 1.06e-14 | 16 | 0 | `ForeachMARSAdamW(lr=0.37652, betas=(0.939, 1.0000), shampoo_beta=1.000)` |
| noisy_matmul-trivial | STORM_plus | No | No | ✗ | 8483.20s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | AdamW | No | Yes | ✗ | 43825.35s | 1.11e-11 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | AdamW | No | No | ✓ | 35154.12s | 9.99e-05 | 26 | 0 | `ForeachAdamW(lr=1.21603, betas=(0.961, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-trivial | MARSAdamW | No | No | ✓ | 35208.12s | 9.92e-05 | 14 | 0 | `ForeachMARSAdamW(lr=4.27505, betas=(0.993, 0.3415), shampoo_beta=0.997)` |
| parameter_scale-trivial | STORM_plus | No | No | ✗ | 8545.02s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | AdamW | No | Yes | ✓ | 38633.64s | 9.59e-05 | 16 | 0 | `ForeachAdamW(lr=3.43573, betas=(0.994, 0.9687), shampoo_beta=1.000)` |
| plateau_navigation-trivial | AdamW | No | No | ✓ | 16838.33s | 9.96e-05 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | MARSAdamW | No | No | ✓ | 21893.20s | 9.99e-05 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | STORM_plus | No | No | ✗ | 8438.73s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| plateau_navigation-trivial | AdamW | No | Yes | ✓ | 17524.64s | 9.96e-05 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | AdamW | No | No | ✗ | 30800.80s | 2.55e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-trivial | MARSAdamW | No | No | ✓ | 25730.72s | 5.99e-12 | 6 | 0 | `ForeachMARSAdamW(lr=7.32095, betas=(0.997, 0.9988), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | STORM_plus | No | No | ✗ | 8077.90s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-trivial | AdamW | No | Yes | ✓ | 43475.29s | 2.14e-14 | 45 | 0 | `ForeachAdamW(lr=0.00171, betas=(0.936, 0.9911), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | AdamW | No | No | ✓ | 44726.22s | 8.88e-16 | 32 | 0 | `ForeachAdamW(lr=0.02326, betas=(0.006, 1.0000), shampoo_beta=0.054)` |
| quadratic_varying_target-trivial | MARSAdamW | No | No | ✓ | 27351.76s | 1.28e-13 | 6 | 0 | `ForeachMARSAdamW(lr=3.33380, betas=(0.986, 0.9828), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | STORM_plus | No | No | ✗ | 8367.14s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_target-trivial | AdamW | No | Yes | ✓ | 35932.57s | 2.67e-15 | 27 | 0 | `ForeachAdamW(lr=0.98099, betas=(0.967, 0.9986), shampoo_beta=0.103)` |
| rastrigin-trivial | AdamW | No | No | ✗ | 42439.80s | 3.98e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | MARSAdamW | No | No | ✗ | 44499.09s | 3.98e+00 | 50 | 0 | `ForeachMARSAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | STORM_plus | No | No | ✗ | 7997.77s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | AdamW | No | Yes | ✗ | 43699.96s | 3.98e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | AdamW | No | No | ✓ | 23735.41s | 9.98e-10 | 6 | 0 | `ForeachAdamW(lr=0.03076, betas=(0.972, 0.9996), shampoo_beta=0.892)` |
| rosenbrock-trivial | MARSAdamW | No | No | ✓ | 27283.71s | 9.41e-10 | 7 | 0 | `ForeachMARSAdamW(lr=0.09657, betas=(0.819, 0.9999), shampoo_beta=0.772)` |
| rosenbrock-trivial | STORM_plus | No | No | ✗ | 8036.69s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | AdamW | No | Yes | ✓ | 25570.91s | 9.96e-10 | 9 | 0 | `ForeachAdamW(lr=0.00465, betas=(0.950, 0.9995), shampoo_beta=0.999)` |
| saddle_point-trivial | AdamW | No | No | ✓ | 21158.57s | 9.98e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | MARSAdamW | No | No | ✓ | 24446.41s | 9.99e-02 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | STORM_plus | No | No | ✗ | 8483.88s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| saddle_point-trivial | AdamW | No | Yes | ✓ | 21398.84s | 9.98e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | AdamW | No | No | ✓ | 23558.69s | 8.05e-04 | 8 | 0 | `ForeachAdamW(lr=0.03494, betas=(0.873, 1.0000), shampoo_beta=0.002)` |
| scale_invariant-trivial | MARSAdamW | No | No | ✓ | 27505.45s | 9.75e-04 | 6 | 0 | `ForeachMARSAdamW(lr=2.62831, betas=(0.998, 0.9996), shampoo_beta=0.999)` |
| scale_invariant-trivial | STORM_plus | No | No | ✗ | 8508.05s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| scale_invariant-trivial | AdamW | No | Yes | ✓ | 23765.30s | 9.94e-04 | 6 | 0 | `ForeachAdamW(lr=0.03245, betas=(0.997, 1.0000), shampoo_beta=0.292)` |
| sparse_gradient-trivial | AdamW | No | No | ✓ | 517.74s | 9.66e-05 | 6 | 0 | `ForeachAdamW(lr=0.00425, betas=(0.742, 1.0000), shampoo_beta=0.714)` |
| sparse_gradient-trivial | MARSAdamW | No | No | ✓ | 18105.32s | 9.69e-05 | 7 | 0 | `ForeachMARSAdamW(lr=0.01585, betas=(0.837, 1.0000), shampoo_beta=0.023)` |
| sparse_gradient-trivial | STORM_plus | No | No | ✓ | 14371.87s | 1.84e-05 | 3 | 0 | `ForeachSTORM_plus(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| sparse_gradient-trivial | AdamW | No | Yes | ✓ | 836.84s | 9.88e-05 | 8 | 0 | `ForeachAdamW(lr=0.00361, betas=(0.989, 1.0000), shampoo_beta=0.997)` |
| wide_linear-trivial | AdamW | No | No | ✗ | 21656.91s | 7.65e-08 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-trivial | MARSAdamW | No | No | ✗ | 11143.09s | 4.53e-07 | 50 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-trivial | STORM_plus | No | No | ✗ | 8448.83s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-trivial | AdamW | No | Yes | ✗ | 26612.50s | 8.40e-08 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-trivial | AdamW | No | No | ✓ | 3198.85s | 9.78e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | MARSAdamW | No | No | ✓ | 3489.18s | 9.92e-04 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | STORM_plus | No | No | ✗ | 8427.93s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-trivial | AdamW | No | Yes | ✓ | 3998.64s | 9.82e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamW | No | No | ✓ | 22033.82s | 9.86e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | MARSAdamW | No | No | ✓ | 28321.00s | 9.60e-04 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | STORM_plus | No | No | ✗ | 8280.15s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-trivial | AdamW | No | Yes | ✓ | 25112.26s | 9.91e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamW | No | No | ✓ | 25473.50s | 9.33e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | MARSAdamW | No | No | ✓ | 32150.98s | 9.90e-03 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | STORM_plus | No | No | ✗ | 8333.15s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence-trivial | AdamW | No | Yes | ✓ | 28407.19s | 9.34e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamW | No | No | ✓ | 25881.18s | 9.93e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | MARSAdamW | No | No | ✓ | 31886.63s | 9.94e-03 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | STORM_plus | No | No | ✗ | 8550.23s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence_rnn-trivial | AdamW | No | Yes | ✓ | 28918.15s | 9.94e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamW | No | No | ✓ | 572.40s | 9.98e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot-trivial | MARSAdamW | No | No | ✓ | 26220.69s | 9.67e-03 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | STORM_plus | No | No | ✗ | 8425.05s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot-trivial | AdamW | No | Yes | ✓ | 732.07s | 9.29e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamW | No | No | ✓ | 9243.79s | 9.61e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | MARSAdamW | No | No | ✓ | 15850.25s | 9.93e-03 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | STORM_plus | No | No | ✗ | 8340.47s | inf | 50 | 0 | `ForeachSTORM_plus(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-trivial | AdamW | No | Yes | ✓ | 10746.13s | 9.87e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors

