# Benchmark Results
Generated: 2025-06-09 17:35:48.467517
Last updated: 2025-06-09 17:35:48.467529

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| MARSAdamW | No | No | 1/1 | 3129.09s | 10.0 |
| AdamW | No | No | 4/4 | 1308.36s | 19.2 |
| AdamW | No | Yes | 3/3 | 786.96s | 12.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| noisy_matmul-medium | AdamW | No | No | ✓ | 46.03s | 9.46e-17 | 6 | 0 | `ForeachAdamW(lr=0.14392, betas=(0.924, 1.0000), shampoo_beta=0.906)` |
| noisy_matmul-medium | AdamW | No | Yes | ✓ | 864.90s | 8.02e-17 | 18 | 0 | `ForeachAdamW(lr=0.18313, betas=(0.816, 1.0000), shampoo_beta=0.895)` |
| xor_sequence-medium | AdamW | No | No | ✓ | 692.06s | 9.17e-03 | 10 | 0 | `ForeachAdamW(lr=0.00962, betas=(0.939, 1.0000), shampoo_beta=0.705)` |
| xor_sequence-medium | MARSAdamW | No | No | ✓ | 3129.09s | 8.77e-03 | 10 | 0 | `ForeachMARSAdamW(lr=0.02595, betas=(0.991, 1.0000), shampoo_beta=0.335)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 451.21s | 9.99e-03 | 6 | 0 | `ForeachAdamW(lr=0.01099, betas=(0.844, 1.0000), shampoo_beta=0.822)` |
| xor_sequence_rnn-medium | AdamW | No | No | ✓ | 1202.87s | 9.80e-03 | 16 | 0 | `ForeachAdamW(lr=0.00120, betas=(0.736, 1.0000), shampoo_beta=0.999)` |
| xor_sequence_rnn-medium | AdamW | No | Yes | ✓ | 1044.77s | 9.85e-03 | 12 | 0 | `ForeachAdamW(lr=0.00374, betas=(0.178, 1.0000), shampoo_beta=0.016)` |
| xor_spot-medium | AdamW | No | No | ✓ | 3292.49s | 9.47e-03 | 45 | 0 | `ForeachAdamW(lr=0.00287, betas=(0.262, 1.0000), shampoo_beta=0.590)` |
