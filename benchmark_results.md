# Benchmark Results
Generated: 2025-06-04 22:07:34.484386
Last updated: 2025-06-04 22:07:34.484393

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SFAdamW | No | Yes | 7/9 | 11926.78s | 7.1 |
| SFAdamWEMA | No | Yes | 7/9 | 11801.32s | 7.1 |
| SFAdamW | No | No | 6/9 | 11697.09s | 7.7 |
| AdamW | No | No | 2/9 | 4982.55s | 6.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| beale-trivial | AdamW | No | No | ✗ | 11382.49s | 1.32e+01 | 15 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| beale-trivial | SFAdamW | No | No | ✓ | 14470.21s | 9.21e-09 | 13 | 0 | `ForeachSFAdamW(lr=64.21445, betas=(0.006, 0.9975), shampoo_beta=0.830)` |
| beale-trivial | SFAdamW | No | Yes | ✓ | 13058.47s | 6.73e-09 | 7 | 0 | `ForeachSFAdamW(lr=99.04932, betas=(0.989, 0.9604), shampoo_beta=0.757)` |
| beale-trivial | SFAdamWEMA | No | Yes | ✓ | 14750.39s | 6.73e-09 | 7 | 0 | `ForeachSFAdamWEMA(lr=99.04932, betas=(0.989, 0.9604), shampoo_beta=0.757)` |
| layer_wise_scale-trivial | AdamW | No | No | ✗ | 8757.71s | 1.17e-01 | 15 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| layer_wise_scale-trivial | SFAdamW | No | No | ✓ | 13529.37s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| layer_wise_scale-trivial | SFAdamW | No | Yes | ✓ | 14314.31s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| layer_wise_scale-trivial | SFAdamWEMA | No | Yes | ✓ | 14301.42s | 1.00e-04 | 4 | 0 | `ForeachSFAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| noisy_matmul-trivial | AdamW | No | No | ✗ | 5426.57s | 1.51e-01 | 15 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | SFAdamW | No | No | ✗ | 9288.34s | 1.42e-13 | 15 | 0 | `ForeachSFAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-trivial | SFAdamW | No | Yes | ✓ | 9587.77s | 1.24e-14 | 11 | 0 | `ForeachSFAdamW(lr=10.27253, betas=(0.997, 1.0000), shampoo_beta=0.729)` |
| noisy_matmul-trivial | SFAdamWEMA | No | Yes | ✓ | 8696.08s | 1.24e-14 | 11 | 0 | `ForeachSFAdamWEMA(lr=10.27253, betas=(0.997, 1.0000), shampoo_beta=0.729)` |
| parameter_scale-trivial | AdamW | No | No | ✗ | 16025.29s | 1.29e-02 | 15 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | SFAdamW | No | No | ✗ | 16335.96s | 8.24e-02 | 15 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | SFAdamW | No | Yes | ✗ | 15559.73s | 2.19e-01 | 15 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-trivial | SFAdamWEMA | No | Yes | ✗ | 16191.15s | 2.19e-01 | 15 | 0 | `ForeachSFAdamWEMA(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-trivial | AdamW | No | No | ✗ | 15075.16s | 2.55e-02 | 15 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-trivial | SFAdamW | No | No | ✓ | 12984.42s | 6.01e-14 | 10 | 0 | `ForeachSFAdamW(lr=0.89992, betas=(0.985, 0.9999), shampoo_beta=0.971)` |
| quadratic_varying_scale-trivial | SFAdamW | No | Yes | ✓ | 15055.39s | 1.18e-14 | 10 | 0 | `ForeachSFAdamW(lr=0.26674, betas=(0.470, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | SFAdamWEMA | No | Yes | ✓ | 15059.94s | 1.18e-14 | 10 | 0 | `ForeachSFAdamWEMA(lr=0.26674, betas=(0.470, 1.0000), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | AdamW | No | No | ✗ | 14988.19s | 1.22e-01 | 15 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_target-trivial | SFAdamW | No | No | ✓ | 15070.21s | 5.33e-15 | 9 | 0 | `ForeachSFAdamW(lr=76.32257, betas=(1.000, 1.0000), shampoo_beta=0.211)` |
| quadratic_varying_target-trivial | SFAdamW | No | Yes | ✓ | 15097.61s | 7.11e-15 | 8 | 0 | `ForeachSFAdamW(lr=61.71814, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| quadratic_varying_target-trivial | SFAdamWEMA | No | Yes | ✓ | 13358.79s | 7.11e-15 | 8 | 0 | `ForeachSFAdamWEMA(lr=61.71814, betas=(1.000, 1.0000), shampoo_beta=0.995)` |
| rastrigin-trivial | AdamW | No | No | ✗ | 7050.80s | 4.01e+00 | 15 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SFAdamW | No | No | ✗ | 16286.87s | 3.98e+00 | 15 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SFAdamW | No | Yes | ✗ | 16128.35s | 3.98e+00 | 15 | 0 | `ForeachSFAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SFAdamWEMA | No | Yes | ✗ | 16125.69s | 3.98e+00 | 15 | 0 | `ForeachSFAdamWEMA(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | AdamW | No | No | ✓ | 9754.75s | 9.98e-10 | 6 | 0 | `ForeachAdamW(lr=0.03076, betas=(0.972, 0.9996), shampoo_beta=0.892)` |
| rosenbrock-trivial | SFAdamW | No | No | ✓ | 12193.66s | 6.08e-10 | 6 | 0 | `ForeachSFAdamW(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| rosenbrock-trivial | SFAdamW | No | Yes | ✓ | 14350.51s | 2.22e-10 | 6 | 0 | `ForeachSFAdamW(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| rosenbrock-trivial | SFAdamWEMA | No | Yes | ✓ | 14412.91s | 2.22e-10 | 6 | 0 | `ForeachSFAdamWEMA(lr=99.64541, betas=(0.960, 0.9953), shampoo_beta=0.996)` |
| sparse_gradient-trivial | AdamW | No | No | ✓ | 210.34s | 9.66e-05 | 6 | 0 | `ForeachAdamW(lr=0.00425, betas=(0.742, 1.0000), shampoo_beta=0.714)` |
| sparse_gradient-trivial | SFAdamW | No | No | ✓ | 1934.65s | 9.99e-05 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | SFAdamW | No | Yes | ✓ | 2023.39s | 1.00e-04 | 4 | 0 | `ForeachSFAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-trivial | SFAdamWEMA | No | Yes | ✓ | 2029.73s | 1.00e-04 | 4 | 0 | `ForeachSFAdamWEMA(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors

