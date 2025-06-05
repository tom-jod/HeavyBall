# Benchmark Results
Generated: 2025-05-23 16:24:49.451301
Last updated: 2025-05-23 16:24:49.451307

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamWEMA | No | Yes | 23/28 | 5607.74s | 3.7 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | AdamWEMA | No | Yes | ✗ | 1247.05s | -4.53e-02 | 50 | 0 | `ForeachAdamWEMA(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamWEMA | No | Yes | ✓ | 440.27s | 1.22e-16 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| beale-trivial | AdamWEMA | No | Yes | ✓ | 10379.66s | 1.00e-08 | 6 | 0 | `ForeachAdamWEMA(lr=0.14123, betas=(0.999, 1.0000), shampoo_beta=0.773)` |
| constrained_optimization-trivial | AdamWEMA | No | Yes | ✓ | 3781.94s | 1.00e+00 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | AdamWEMA | No | Yes | ✓ | 9436.57s | 1.05e-08 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| dynamic_landscape-trivial | AdamWEMA | No | Yes | ✓ | 486.57s | 9.70e-03 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_delay-trivial | AdamWEMA | No | Yes | ✓ | 222.67s | 9.65e-05 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| gradient_noise_scale-trivial | AdamWEMA | No | Yes | ✓ | 5459.42s | 1.61e-06 | 6 | 0 | `ForeachAdamWEMA(lr=0.00958, betas=(1.000, 1.0000), shampoo_beta=0.518)` |
| layer_wise_scale-trivial | AdamWEMA | No | Yes | ✓ | 5426.27s | 9.73e-05 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| minimax-trivial | AdamWEMA | No | Yes | ✗ | 7749.64s | 2.41e+00 | 50 | 0 | `ForeachAdamWEMA(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-trivial | AdamWEMA | No | Yes | ✓ | 4786.57s | -3.87e-06 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| noisy_matmul-trivial | AdamWEMA | No | Yes | ✗ | 10927.96s | 6.62e-14 | 50 | 0 | `ForeachAdamWEMA(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | AdamWEMA | No | Yes | ✗ | 15588.11s | 1.77e-01 | 50 | 0 | `ForeachAdamWEMA(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| plateau_navigation-trivial | AdamWEMA | No | Yes | ✓ | 6935.52s | 1.00e-04 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | AdamWEMA | No | Yes | ✓ | 9379.58s | 1.22e-14 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| quadratic_varying_target-trivial | AdamWEMA | No | Yes | ✓ | 9345.29s | 8.88e-16 | 10 | 0 | `ForeachAdamWEMA(lr=0.06032, betas=(0.993, 1.0000), shampoo_beta=0.524)` |
| rastrigin-trivial | AdamWEMA | No | Yes | ✓ | 8513.53s | 7.07e-03 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| rosenbrock-trivial | AdamWEMA | No | Yes | ✓ | 5122.78s | 1.00e-09 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| saddle_point-trivial | AdamWEMA | No | Yes | ✓ | 6439.63s | 9.99e-02 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | AdamWEMA | No | Yes | ✓ | 8312.13s | 9.47e-04 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | AdamWEMA | No | Yes | ✓ | 41.91s | 9.97e-05 | 4 | 0 | `ForeachAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| wide_linear-trivial | AdamWEMA | No | Yes | ✗ | 9009.25s | 7.00e-08 | 50 | 0 | `ForeachAdamWEMA(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-trivial | AdamWEMA | No | Yes | ✓ | 762.81s | 9.60e-04 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamWEMA | No | Yes | ✓ | 9358.01s | 9.68e-04 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamWEMA | No | Yes | ✓ | 9900.81s | 9.95e-03 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamWEMA | No | Yes | ✓ | 9797.01s | 9.98e-03 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamWEMA | No | Yes | ✓ | 500.08s | 9.87e-03 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot_rnn-trivial | AdamWEMA | No | Yes | ✓ | 4148.99s | 9.96e-03 | 2 | 0 | `ForeachAdamWEMA(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |

## Errors

