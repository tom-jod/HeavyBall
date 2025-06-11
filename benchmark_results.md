# Benchmark Results
Generated: 2025-06-11 09:55:35.702789
Last updated: 2025-06-11 09:55:35.702794

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| MARSAdamW | No | No | 4/4 | 278.11s | 15.5 |
| AdamW | No | No | 4/4 | 312.58s | 19.2 |
| AdamW | No | Yes | 4/4 | 340.16s | 19.5 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| noisy_matmul-medium | AdamW | No | No | ✓ | 45.73s | 9.46e-17 | 6 | 0 | `ForeachAdamW(lr=0.14392, betas=(0.924, 1.0000), shampoo_beta=0.906)` |
| noisy_matmul-medium | MARSAdamW | No | No | ✓ | 88.60s | 1.83e-16 | 7 | 0 | `ForeachMARSAdamW(lr=55.43027, betas=(0.997, 1.0000), shampoo_beta=0.017)` |
| noisy_matmul-medium | AdamW | No | Yes | ✓ | 326.31s | 8.02e-17 | 18 | 0 | `ForeachAdamW(lr=0.18313, betas=(0.816, 1.0000), shampoo_beta=0.895)` |
| xor_sequence-medium | AdamW | No | No | ✓ | 221.58s | 9.17e-03 | 10 | 0 | `ForeachAdamW(lr=0.00962, betas=(0.939, 1.0000), shampoo_beta=0.705)` |
| xor_sequence-medium | MARSAdamW | No | No | ✓ | 184.01s | 9.41e-03 | 4 | 0 | `ForeachMARSAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 198.36s | 9.99e-03 | 6 | 0 | `ForeachAdamW(lr=0.01099, betas=(0.844, 1.0000), shampoo_beta=0.822)` |
| xor_sequence_rnn-medium | AdamW | No | No | ✓ | 356.81s | 9.80e-03 | 16 | 0 | `ForeachAdamW(lr=0.00120, betas=(0.736, 1.0000), shampoo_beta=0.999)` |
| xor_sequence_rnn-medium | MARSAdamW | No | No | ✓ | 112.64s | 7.37e-03 | 4 | 0 | `ForeachMARSAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_sequence_rnn-medium | AdamW | No | Yes | ✓ | 257.83s | 9.85e-03 | 12 | 0 | `ForeachAdamW(lr=0.00374, betas=(0.178, 1.0000), shampoo_beta=0.016)` |
| xor_spot-medium | AdamW | No | No | ✓ | 626.19s | 9.47e-03 | 45 | 0 | `ForeachAdamW(lr=0.00287, betas=(0.262, 1.0000), shampoo_beta=0.590)` |
| xor_spot-medium | MARSAdamW | No | No | ✓ | 727.20s | 7.26e-03 | 47 | 0 | `ForeachMARSAdamW(lr=0.01222, betas=(0.927, 0.9999), shampoo_beta=0.893)` |
| xor_spot-medium | AdamW | No | Yes | ✓ | 578.14s | 8.26e-03 | 42 | 0 | `ForeachAdamW(lr=0.00350, betas=(0.488, 1.0000), shampoo_beta=0.011)` |
