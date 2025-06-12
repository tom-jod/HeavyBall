# Benchmark Results
Generated: 2025-06-12 17:42:50.330712
Last updated: 2025-06-12 17:42:50.330725

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 1/2 | 250.64s | 3.0 |
| MARSAdamW | No | No | 1/2 | 1168.61s | 7.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| batch_size_scaling-medium | MARSAdamW | No | No | ✓ | 1168.61s | 1.09e-16 | 7 | 0 | `ForeachMARSAdamW(lr=0.02179, betas=(0.992, 0.9999), shampoo_beta=0.056)` |
| batch_size_scaling-medium | SGD | No | No | ✓ | 250.64s | 1.04e-16 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| gradient_delay-medium | MARSAdamW | No | No | ✗ | 1.98s | inf | 200 | 0 | N/A |
| gradient_delay-medium | SGD | No | No | ✗ | 2.14s | inf | 200 | 0 | N/A |

## Errors


### gradient_delay-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 580, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=2000)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 580, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=2000)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```
