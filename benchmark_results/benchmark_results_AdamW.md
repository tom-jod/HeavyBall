# Benchmark Results
Generated: 2025-05-21 16:36:19.933499
Last updated: 2025-05-21 16:36:19.933505

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamW | No | No | 21/28 | 5242.00s | 11.4 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | AdamW | No | No | ✗ | 1602.13s | -4.58e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamW | No | No | ✓ | 888.03s | 1.08e-16 | 6 | 0 | `ForeachAdamW(lr=0.02090, betas=(0.985, 1.0000), shampoo_beta=0.394)` |
| beale-trivial | AdamW | No | No | ✗ | 6853.06s | 1.31e+01 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-trivial | AdamW | No | No | ✓ | 4419.63s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | AdamW | No | No | ✗ | 11231.41s | 5.17e-07 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | AdamW | No | No | ✓ | 3305.39s | 9.61e-03 | 6 | 0 | `ForeachAdamW(lr=0.46575, betas=(0.959, 1.0000), shampoo_beta=0.936)` |
| gradient_delay-trivial | AdamW | No | No | ✓ | 369.97s | 9.53e-05 | 7 | 0 | `ForeachAdamW(lr=0.02505, betas=(0.944, 1.0000), shampoo_beta=0.649)` |
| gradient_noise_scale-trivial | AdamW | No | No | ✓ | 6480.66s | 1.00e-06 | 31 | 0 | `ForeachAdamW(lr=0.00066, betas=(0.999, 1.0000), shampoo_beta=0.612)` |
| layer_wise_scale-trivial | AdamW | No | No | ✓ | 6021.99s | 1.00e-04 | 22 | 0 | `ForeachAdamW(lr=0.00054, betas=(1.000, 1.0000), shampoo_beta=0.728)` |
| minimax-trivial | AdamW | No | No | ✗ | 7107.74s | 1.58e+00 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-trivial | AdamW | No | No | ✓ | 7018.67s | 9.48e-07 | 22 | 0 | `ForeachAdamW(lr=0.00167, betas=(0.997, 1.0000), shampoo_beta=0.857)` |
| noisy_matmul-trivial | AdamW | No | No | ✓ | 8208.87s | 1.42e-14 | 46 | 0 | `ForeachAdamW(lr=6.51946, betas=(0.993, 1.0000), shampoo_beta=0.671)` |
| parameter_scale-trivial | AdamW | No | No | ✓ | 10284.19s | 9.99e-05 | 26 | 0 | `ForeachAdamW(lr=1.21603, betas=(0.961, 1.0000), shampoo_beta=1.000)` |
| plateau_navigation-trivial | AdamW | No | No | ✓ | 5181.49s | 9.96e-05 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | AdamW | No | No | ✗ | 9618.64s | 2.55e-02 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_target-trivial | AdamW | No | No | ✓ | 13813.31s | 8.88e-16 | 32 | 0 | `ForeachAdamW(lr=0.02326, betas=(0.006, 1.0000), shampoo_beta=0.054)` |
| rastrigin-trivial | AdamW | No | No | ✗ | 12792.62s | 3.98e+00 | 50 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | AdamW | No | No | ✓ | 5983.45s | 9.98e-10 | 6 | 0 | `ForeachAdamW(lr=0.03076, betas=(0.972, 0.9996), shampoo_beta=0.892)` |
| saddle_point-trivial | AdamW | No | No | ✓ | 6907.80s | 9.98e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | AdamW | No | No | ✓ | 5779.87s | 8.05e-04 | 8 | 0 | `ForeachAdamW(lr=0.03494, betas=(0.873, 1.0000), shampoo_beta=0.002)` |
| sparse_gradient-trivial | AdamW | No | No | ✓ | 149.81s | 9.66e-05 | 6 | 0 | `ForeachAdamW(lr=0.00425, betas=(0.742, 1.0000), shampoo_beta=0.714)` |
| wide_linear-trivial | AdamW | No | No | ✗ | 7231.16s | 7.65e-08 | 50 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-trivial | AdamW | No | No | ✓ | 1064.15s | 9.78e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamW | No | No | ✓ | 5611.16s | 9.86e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamW | No | No | ✓ | 8022.76s | 9.33e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamW | No | No | ✓ | 7962.09s | 9.93e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamW | No | No | ✓ | 235.90s | 9.98e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamW | No | No | ✓ | 2372.73s | 9.61e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors

