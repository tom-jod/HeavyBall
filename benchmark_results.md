# Benchmark Results
Generated: 2025-06-12 12:58:56.992577
Last updated: 2025-06-12 12:58:56.992589

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 15/16 | 2102.74s | 7.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | SGD | No | No | ✗ | 1229.88s | -4.57e-02 | 50 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | SGD | No | No | ✓ | 175.15s | 1.62e-16 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| constrained_optimization-trivial | SGD | No | No | ✓ | 4163.16s | 1.01e+00 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| dynamic_landscape-trivial | SGD | No | No | ✓ | 421.81s | 9.94e-03 | 7 | 0 | `ForeachSGD(lr=99.33942, betas=(1.000, 0.0441), shampoo_beta=1.000)` |
| gradient_delay-trivial | SGD | No | No | ✓ | 165.07s | 9.82e-05 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| gradient_noise_scale-trivial | SGD | No | No | ✓ | 2766.23s | 1.10e-06 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| layer_wise_scale-trivial | SGD | No | No | ✓ | 3617.15s | 9.98e-05 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| noisy_matmul-trivial | SGD | No | No | ✓ | 3254.06s | 1.13e-14 | 8 | 0 | `ForeachSGD(lr=10.52959, betas=(0.987, 0.2967), shampoo_beta=0.995)` |
| parameter_scale-trivial | SGD | No | No | ✓ | 3605.45s | 9.99e-05 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| rosenbrock-trivial | SGD | No | No | ✓ | 3468.91s | 9.84e-10 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| saddle_point-trivial | SGD | No | No | ✓ | 3843.56s | 9.89e-02 | 2 | 0 | `ForeachSGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| sparse_gradient-trivial | SGD | No | No | ✓ | 1146.98s | 9.98e-05 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| xor_digit-trivial | SGD | No | No | ✓ | 405.27s | 9.85e-04 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_digit_rnn-trivial | SGD | No | No | ✓ | 575.27s | 9.97e-04 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot-trivial | SGD | No | No | ✓ | 2239.35s | 8.90e-03 | 50 | 0 | `ForeachSGD(lr=0.65703, betas=(0.514, 0.1294), shampoo_beta=1.000)` |
| xor_spot_rnn-trivial | SGD | No | No | ✓ | 1693.64s | 9.94e-03 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors

