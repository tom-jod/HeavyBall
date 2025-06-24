# Benchmark Results
Generated: 2025-06-23 11:31:48.407825
Last updated: 2025-06-23 11:31:48.407838

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 0/1 | 0.00s | 0.0 |
| MARSAdamW | No | No | 0/1 | 0.00s | 0.0 |
| SFAdamW | No | No | 0/1 | 0.00s | 0.0 |
| SFAdamW | No | Yes | 0/1 | 0.00s | 0.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| minimax-hard | MARSAdamW | No | No | ✗ | 6.74s | inf | 100 | 0 | N/A |
| minimax-hard | SFAdamW | No | No | ✗ | 2.69s | inf | 100 | 0 | N/A |
| minimax-hard | SGD | No | No | ✗ | 1.64s | inf | 100 | 0 | N/A |
| minimax-hard | SFAdamW | No | Yes | ✗ | 8.34s | inf | 100 | 0 | N/A |

## Errors


### minimax-hard - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 752, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 8192])) must be the same as input size (torch.Size([]))

```

### minimax-hard - SFAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 752, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 8192])) must be the same as input size (torch.Size([]))

```

### minimax-hard - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 752, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 8192])) must be the same as input size (torch.Size([]))

```

### minimax-hard - mars-SFAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 752, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 8192])) must be the same as input size (torch.Size([]))

```
