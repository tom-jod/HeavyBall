# Benchmark Results
Generated: 2025-06-30 17:18:49.964273
Last updated: 2025-06-30 17:18:49.964278

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 1/1 | 31.31s | 16.0 |
| AdamW | No | No | 0/1 | 0.00s | 0.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| rosenbrock-trivial | AdamW | No | No | ✗ | 41.17s | 1.23e+01 | 18 | 0 | `ForeachAdamW(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | SGD | No | No | ✓ | 31.31s | 9.89e-10 | 16 | 0 | `ForeachSGD(lr=0.08264, betas=(1.000, 0.9965), shampoo_beta=1.000)` |

## Errors

