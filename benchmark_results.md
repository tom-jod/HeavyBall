# Benchmark Results
Generated: 2025-06-09 16:51:48.204521
Last updated: 2025-06-09 16:51:48.204533

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamW | No | No | 2/2 | 369.04s | 8.0 |
| AdamW | No | Yes | 1/1 | 451.21s | 6.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| noisy_matmul-medium | AdamW | No | No | ✓ | 46.03s | 9.46e-17 | 6 | 0 | `ForeachAdamW(lr=0.14392, betas=(0.924, 1.0000), shampoo_beta=0.906)` |
| xor_sequence-medium | AdamW | No | No | ✓ | 692.06s | 9.17e-03 | 10 | 0 | `ForeachAdamW(lr=0.00962, betas=(0.939, 1.0000), shampoo_beta=0.705)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 451.21s | 9.99e-03 | 6 | 0 | `ForeachAdamW(lr=0.01099, betas=(0.844, 1.0000), shampoo_beta=0.822)` |
