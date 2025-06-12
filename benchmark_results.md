# Benchmark Results
Generated: 2025-06-12 17:03:53.491526
Last updated: 2025-06-12 17:03:53.491540

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 0/9 | 0.00s | 0.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| beale-trivial | SGD | No | No | ✗ | 1.13s | 3.39e+04 | 1 | 0 | `ForeachSGD(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-trivial | SGD | No | No | ✗ | 1.03s | 4.00e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-trivial | SGD | No | No | ✗ | 1.10s | 1.49e+00 | 1 | 0 | `ForeachSGD(lr=0.10000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| gradient_delay-trivial | SGD | No | No | ✗ | 1.04s | inf | 1 | 0 | N/A |
| plateau_navigation-trivial | SGD | No | No | ✗ | 0.29s | 7.02e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-trivial | SGD | No | No | ✗ | 1.36s | 1.81e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rastrigin-trivial | SGD | No | No | ✗ | 1.06s | 1.17e+01 | 1 | 0 | `ForeachSGD(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| rosenbrock-trivial | SGD | No | No | ✗ | 1.35s | 2.87e+03 | 1 | 0 | `ForeachSGD(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| saddle_point-trivial | SGD | No | No | ✗ | 1.18s | 1.00e+01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |

## Errors


### gradient_delay-trivial - SGD
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
