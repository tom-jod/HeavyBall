# Benchmark Results
Generated: 2025-06-23 18:30:16.369154
Last updated: 2025-06-23 18:30:16.369160

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 1/24 | 0.49s | 1.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-nightmare | SGD | No | No | ✗ | 1.04s | 1.07e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-nightmare | SGD | No | No | ✗ | 0.56s | 2.06e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| constrained_optimization-nightmare | SGD | No | No | ✗ | 1.01s | 4.00e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-nightmare | SGD | No | No | ✗ | 1.14s | 1.50e+00 | 1 | 0 | `ForeachSGD(lr=0.10000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| gradient_delay-nightmare | SGD | No | No | ✗ | 0.49s | inf | 1 | 0 | N/A |
| gradient_noise_scale-nightmare | SGD | No | No | ✗ | 0.01s | 9.87e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| layer_wise_scale-nightmare | SGD | No | No | ✗ | 0.69s | 3.51e+08 | 1 | 0 | `ForeachSGD(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| minimax-nightmare | SGD | No | No | ✗ | 0.27s | inf | 1 | 0 | N/A |
| momentum_utilization-nightmare | SGD | No | No | ✗ | 0.51s | 1.06e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| noisy_matmul-nightmare | SGD | No | No | ✗ | 0.72s | 3.87e-06 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| parameter_scale-nightmare | SGD | No | No | ✗ | 0.88s | 3.52e+05 | 1 | 0 | `ForeachSGD(lr=0.00010, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| plateau_navigation-nightmare | SGD | No | No | ✗ | 0.70s | 1.00e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_scale-nightmare | SGD | No | No | ✗ | 1.36s | 1.00e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| quadratic_varying_target-nightmare | SGD | No | No | ✗ | 0.97s | 1.00e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| saddle_point-nightmare | SGD | No | No | ✗ | 1.23s | 1.00e+01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| scale_invariant-nightmare | SGD | No | No | ✗ | 0.58s | 6.47e+00 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| sparse_gradient-nightmare | SGD | No | No | ✓ | 0.49s | 4.14e-05 | 1 | 0 | `ForeachSGD(lr=0.00000, betas=(0.998, 1.0000), shampoo_beta=1.000)` |
| wide_linear-nightmare | SGD | No | No | ✗ | 0.70s | 2.43e+04 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-nightmare | SGD | No | No | ✗ | 0.10s | 7.08e-01 | 1 | 0 | `ForeachSGD(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-nightmare | SGD | No | No | ✗ | 0.20s | 7.16e-01 | 1 | 0 | `ForeachSGD(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence-nightmare | SGD | No | No | ✗ | 0.18s | 7.03e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence_rnn-nightmare | SGD | No | No | ✗ | 0.29s | 7.12e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot-nightmare | SGD | No | No | ✗ | 0.17s | 7.58e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-nightmare | SGD | No | No | ✗ | 0.10s | 7.06e-01 | 1 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |

## Errors


### gradient_delay-nightmare - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 768, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 743, in compute_hvp
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

### minimax-nightmare - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 581, in trial
    condition_numbers.append( estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 754, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/miniconda3/envs/heavyball/lib/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 131072])) must be the same as input size (torch.Size([]))

```
