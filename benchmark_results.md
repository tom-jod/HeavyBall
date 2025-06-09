# Benchmark Results
Generated: 2025-06-06 22:10:10.521641
Last updated: 2025-06-06 22:10:10.521646

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| MARSAdamW | No | No | 5/7 | 1892.22s | 37.4 |
| AdamW | No | No | 4/7 | 889.01s | 19.2 |
| AdamW | No | Yes | 4/7 | 982.27s | 19.5 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| noisy_matmul-medium | AdamW | No | No | ✓ | 73.55s | 9.46e-17 | 6 | 0 | `ForeachAdamW(lr=0.14392, betas=(0.924, 1.0000), shampoo_beta=0.906)` |
| noisy_matmul-medium | MARSAdamW | No | No | ✓ | 65.89s | 9.46e-17 | 6 | 0 | `ForeachMARSAdamW(lr=0.14392, betas=(0.924, 1.0000), shampoo_beta=0.906)` |
| noisy_matmul-medium | AdamW | No | Yes | ✓ | 592.60s | 8.02e-17 | 18 | 0 | `ForeachAdamW(lr=0.18313, betas=(0.816, 1.0000), shampoo_beta=0.895)` |
| xor_digit-medium | AdamW | No | No | ✗ | 17360.86s | 6.81e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-medium | MARSAdamW | No | No | ✓ | 3793.84s | 9.84e-04 | 84 | 0 | `ForeachMARSAdamW(lr=0.00626, betas=(0.026, 0.3519), shampoo_beta=0.187)` |
| xor_digit-medium | AdamW | No | Yes | ✗ | 13376.78s | 6.71e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | AdamW | No | No | ✗ | 13360.20s | 6.85e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | MARSAdamW | No | No | ✗ | 16836.87s | 6.83e-01 | 1000 | 0 | `ForeachMARSAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | AdamW | No | Yes | ✗ | 18583.08s | 6.79e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence-medium | AdamW | No | No | ✓ | 609.70s | 9.17e-03 | 10 | 0 | `ForeachAdamW(lr=0.00962, betas=(0.939, 1.0000), shampoo_beta=0.705)` |
| xor_sequence-medium | MARSAdamW | No | No | ✓ | 592.64s | 7.06e-03 | 8 | 0 | `ForeachMARSAdamW(lr=0.02137, betas=(0.732, 1.0000), shampoo_beta=0.754)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 381.21s | 9.99e-03 | 6 | 0 | `ForeachAdamW(lr=0.01099, betas=(0.844, 1.0000), shampoo_beta=0.822)` |
| xor_sequence_rnn-medium | AdamW | No | No | ✓ | 840.61s | 9.80e-03 | 16 | 0 | `ForeachAdamW(lr=0.00120, betas=(0.736, 1.0000), shampoo_beta=0.999)` |
| xor_sequence_rnn-medium | MARSAdamW | No | No | ✓ | 1670.42s | 9.44e-03 | 24 | 0 | `ForeachMARSAdamW(lr=0.00945, betas=(0.455, 1.0000), shampoo_beta=0.760)` |
| xor_sequence_rnn-medium | AdamW | No | Yes | ✓ | 727.50s | 9.85e-03 | 12 | 0 | `ForeachAdamW(lr=0.00374, betas=(0.178, 1.0000), shampoo_beta=0.016)` |
| xor_spot-medium | AdamW | No | No | ✓ | 2032.18s | 9.47e-03 | 45 | 0 | `ForeachAdamW(lr=0.00287, betas=(0.262, 1.0000), shampoo_beta=0.590)` |
| xor_spot-medium | MARSAdamW | No | No | ✓ | 3338.34s | 4.83e-03 | 65 | 0 | `ForeachMARSAdamW(lr=0.00299, betas=(0.051, 0.1643), shampoo_beta=0.141)` |
| xor_spot-medium | AdamW | No | Yes | ✓ | 2227.76s | 8.26e-03 | 42 | 0 | `ForeachAdamW(lr=0.00350, betas=(0.488, 1.0000), shampoo_beta=0.011)` |
| xor_spot_rnn-medium | AdamW | No | No | ✗ | 18410.11s | 6.89e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-medium | MARSAdamW | No | No | ✗ | 19478.60s | 6.92e-01 | 1000 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-medium | AdamW | No | Yes | ✗ | 18739.41s | 6.89e-01 | 1000 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |

## Errors

