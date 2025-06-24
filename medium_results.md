# Benchmark Results
Generated: 2025-06-23 05:56:41.940078
Last updated: 2025-06-23 05:56:41.940093

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| SGD | No | No | 9/24 | 59610.71s | 3.7 |
| MARSAdamW | No | No | 12/24 | 51665.48s | 12.1 |
| AdamW | No | No | 12/24 | 58579.59s | 17.5 |
| AdamW | No | Yes | 13/24 | 56830.69s | 18.3 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-medium | AdamW | No | No | ✓ | 28473.35s | 3.65e-03 | 75 | 0 | `ForeachAdamW(lr=0.10810, betas=(0.971, 0.7644), shampoo_beta=1.000)` |
| adversarial_gradient-medium | MARSAdamW | No | No | ✓ | 1629.41s | 6.52e-03 | 6 | 0 | `ForeachMARSAdamW(lr=0.11045, betas=(0.948, 1.0000), shampoo_beta=0.103)` |
| adversarial_gradient-medium | SGD | No | No | ✓ | 1116.12s | 9.99e-03 | 7 | 0 | `ForeachSGD(lr=69.32837, betas=(0.988, 0.0235), shampoo_beta=1.000)` |
| adversarial_gradient-medium | AdamW | No | Yes | ✓ | 12539.98s | 7.75e-03 | 17 | 0 | `ForeachAdamW(lr=0.01942, betas=(0.842, 0.9970), shampoo_beta=0.282)` |
| batch_size_scaling-medium | AdamW | No | No | ✓ | 2213.98s | 1.85e-16 | 6 | 0 | `ForeachAdamW(lr=0.01780, betas=(0.993, 1.0000), shampoo_beta=0.952)` |
| batch_size_scaling-medium | MARSAdamW | No | No | ✓ | 1537.56s | 1.13e-16 | 6 | 0 | `ForeachMARSAdamW(lr=0.03476, betas=(0.932, 1.0000), shampoo_beta=0.521)` |
| batch_size_scaling-medium | SGD | No | No | ✓ | 1023.69s | 1.53e-16 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| batch_size_scaling-medium | AdamW | No | Yes | ✓ | 1596.16s | 1.34e-16 | 6 | 0 | `ForeachAdamW(lr=0.02065, betas=(0.982, 1.0000), shampoo_beta=0.359)` |
| constrained_optimization-medium | AdamW | No | No | ✓ | 51707.90s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | MARSAdamW | No | No | ✓ | 56358.46s | 1.00e+00 | 2 | 0 | `ForeachMARSAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | SGD | No | No | ✓ | 45394.33s | 1.00e+00 | 2 | 0 | `ForeachSGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-medium | AdamW | No | Yes | ✓ | 48973.69s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| dynamic_landscape-medium | AdamW | No | No | ✓ | 27045.33s | 9.85e-03 | 6 | 0 | `ForeachAdamW(lr=4.53671, betas=(0.977, 0.9701), shampoo_beta=0.997)` |
| dynamic_landscape-medium | MARSAdamW | No | No | ✓ | 33529.16s | 9.94e-03 | 3 | 0 | `ForeachMARSAdamW(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| dynamic_landscape-medium | SGD | No | No | ✗ | 50001.48s | 4.82e-01 | 100 | 0 | `ForeachSGD(lr=0.10000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| dynamic_landscape-medium | AdamW | No | Yes | ✓ | 9471.69s | 9.80e-03 | 12 | 0 | `ForeachAdamW(lr=5.57468, betas=(0.950, 1.0000), shampoo_beta=0.999)` |
| gradient_delay-medium | AdamW | No | No | ✗ | 0.55s | inf | 100 | 0 | N/A |
| gradient_delay-medium | MARSAdamW | No | No | ✗ | 0.35s | inf | 100 | 0 | N/A |
| gradient_delay-medium | SGD | No | No | ✗ | 0.20s | inf | 100 | 0 | N/A |
| gradient_delay-medium | AdamW | No | Yes | ✗ | 0.21s | inf | 100 | 0 | N/A |
| gradient_noise_scale-medium | AdamW | No | No | ✓ | 69098.19s | 1.00e-06 | 23 | 0 | `ForeachAdamW(lr=0.00049, betas=(1.000, 1.0000), shampoo_beta=0.585)` |
| gradient_noise_scale-medium | MARSAdamW | No | No | ✓ | 85519.95s | 1.13e-06 | 30 | 0 | `ForeachMARSAdamW(lr=0.06448, betas=(0.971, 0.8679), shampoo_beta=0.999)` |
| gradient_noise_scale-medium | SGD | No | No | ✓ | 72191.99s | 1.08e-06 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| gradient_noise_scale-medium | AdamW | No | Yes | ✓ | 66721.33s | 1.03e-06 | 17 | 0 | `ForeachAdamW(lr=0.00476, betas=(0.998, 1.0000), shampoo_beta=0.781)` |
| layer_wise_scale-medium | AdamW | No | No | ✓ | 134338.42s | 1.00e-04 | 20 | 0 | `ForeachAdamW(lr=0.00056, betas=(1.000, 1.0000), shampoo_beta=0.832)` |
| layer_wise_scale-medium | MARSAdamW | No | No | ✓ | 154541.80s | 9.67e-05 | 21 | 0 | `ForeachMARSAdamW(lr=3.49287, betas=(0.999, 0.9515), shampoo_beta=0.999)` |
| layer_wise_scale-medium | SGD | No | No | ✗ | 156341.32s | inf | 100 | 0 | N/A |
| layer_wise_scale-medium | AdamW | No | Yes | ✓ | 122315.20s | 1.00e-04 | 10 | 0 | `ForeachAdamW(lr=0.01021, betas=(0.997, 1.0000), shampoo_beta=0.954)` |
| minimax-medium | AdamW | No | No | ✗ | 0.00s | inf | 100 | 0 | N/A |
| minimax-medium | MARSAdamW | No | No | ✗ | 1.61s | inf | 100 | 0 | N/A |
| minimax-medium | SGD | No | No | ✗ | 3.16s | inf | 100 | 0 | N/A |
| minimax-medium | AdamW | No | Yes | ✗ | 0.00s | inf | 100 | 0 | N/A |
| momentum_utilization-medium | AdamW | No | No | ✗ | 155495.15s | inf | 100 | 0 | N/A |
| momentum_utilization-medium | MARSAdamW | No | No | ✗ | 159355.58s | inf | 100 | 0 | N/A |
| momentum_utilization-medium | SGD | No | No | ✓ | 114090.03s | -1.30e-03 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| momentum_utilization-medium | AdamW | No | Yes | ✗ | 155332.53s | inf | 100 | 0 | N/A |
| noisy_matmul-medium | AdamW | No | No | ✗ | 160033.51s | inf | 100 | 0 | N/A |
| noisy_matmul-medium | MARSAdamW | No | No | ✗ | 158397.71s | inf | 100 | 0 | N/A |
| noisy_matmul-medium | SGD | No | No | ✗ | 157258.24s | inf | 100 | 0 | N/A |
| noisy_matmul-medium | AdamW | No | Yes | ✗ | 156330.73s | inf | 100 | 0 | N/A |
| parameter_scale-medium | AdamW | No | No | ✓ | 119404.33s | 9.95e-05 | 30 | 0 | `ForeachAdamW(lr=3.57852, betas=(0.990, 1.0000), shampoo_beta=1.000)` |
| parameter_scale-medium | MARSAdamW | No | No | ✓ | 137886.39s | 9.83e-05 | 10 | 0 | `ForeachMARSAdamW(lr=0.44253, betas=(0.983, 0.9978), shampoo_beta=0.999)` |
| parameter_scale-medium | SGD | No | No | ✓ | 137274.34s | 9.99e-05 | 3 | 0 | `ForeachSGD(lr=4.56237, betas=(0.949, 0.6665), shampoo_beta=0.999)` |
| parameter_scale-medium | AdamW | No | Yes | ✓ | 119739.28s | 9.95e-05 | 52 | 0 | `ForeachAdamW(lr=0.65470, betas=(0.937, 0.9989), shampoo_beta=1.000)` |
| plateau_navigation-medium | AdamW | No | No | ✗ | 174397.30s | inf | 100 | 0 | N/A |
| plateau_navigation-medium | MARSAdamW | No | No | ✗ | 166391.84s | inf | 100 | 0 | N/A |
| plateau_navigation-medium | SGD | No | No | ✗ | 183488.79s | inf | 100 | 0 | N/A |
| plateau_navigation-medium | AdamW | No | Yes | ✗ | 173579.62s | inf | 100 | 0 | N/A |
| quadratic_varying_scale-medium | AdamW | No | No | ✗ | 166781.99s | inf | 100 | 0 | N/A |
| quadratic_varying_scale-medium | MARSAdamW | No | No | ✗ | 178731.44s | inf | 100 | 0 | N/A |
| quadratic_varying_scale-medium | SGD | No | No | ✗ | 163033.15s | inf | 100 | 0 | N/A |
| quadratic_varying_scale-medium | AdamW | No | Yes | ✗ | 165668.51s | inf | 100 | 0 | N/A |
| quadratic_varying_target-medium | AdamW | No | No | ✗ | 166828.50s | inf | 100 | 0 | N/A |
| quadratic_varying_target-medium | MARSAdamW | No | No | ✗ | 178810.84s | inf | 100 | 0 | N/A |
| quadratic_varying_target-medium | SGD | No | No | ✗ | 164101.20s | inf | 100 | 0 | N/A |
| quadratic_varying_target-medium | AdamW | No | Yes | ✗ | 165639.28s | inf | 100 | 0 | N/A |
| saddle_point-medium | AdamW | No | No | ✓ | 136945.38s | 9.99e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | MARSAdamW | No | No | ✗ | 162478.48s | inf | 100 | 0 | N/A |
| saddle_point-medium | SGD | No | No | ✓ | 138253.29s | 9.95e-02 | 2 | 0 | `ForeachSGD(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-medium | AdamW | No | Yes | ✓ | 148579.09s | 9.99e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-medium | AdamW | No | No | ✗ | 170514.04s | inf | 100 | 0 | N/A |
| scale_invariant-medium | MARSAdamW | No | No | ✗ | 185701.42s | inf | 100 | 0 | N/A |
| scale_invariant-medium | SGD | No | No | ✗ | 173739.38s | inf | 100 | 0 | N/A |
| scale_invariant-medium | AdamW | No | Yes | ✗ | 173011.42s | inf | 100 | 0 | N/A |
| sparse_gradient-medium | AdamW | No | No | ✓ | 2221.06s | 9.91e-05 | 7 | 0 | `ForeachAdamW(lr=0.00445, betas=(0.999, 1.0000), shampoo_beta=1.000)` |
| sparse_gradient-medium | MARSAdamW | No | No | ✓ | 2730.27s | 9.84e-05 | 4 | 0 | `ForeachMARSAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| sparse_gradient-medium | SGD | No | No | ✓ | 24800.40s | 1.00e-04 | 6 | 0 | `ForeachSGD(lr=15.03653, betas=(0.940, 0.1201), shampoo_beta=0.999)` |
| sparse_gradient-medium | AdamW | No | Yes | ✓ | 4262.42s | 9.79e-05 | 6 | 0 | `ForeachAdamW(lr=0.00202, betas=(1.000, 1.0000), shampoo_beta=0.246)` |
| wide_linear-medium | AdamW | No | No | ✗ | 118277.58s | 6.56e-06 | 100 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-medium | MARSAdamW | No | No | ✗ | 125533.37s | 6.79e-06 | 100 | 0 | `ForeachMARSAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-medium | SGD | No | No | ✗ | 28574.78s | 1.01e-05 | 100 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| wide_linear-medium | AdamW | No | Yes | ✗ | 149384.81s | 6.84e-06 | 100 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-medium | AdamW | No | No | ✗ | 146523.28s | 6.85e-01 | 100 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-medium | MARSAdamW | No | No | ✓ | 38414.06s | 7.67e-04 | 25 | 0 | `ForeachMARSAdamW(lr=0.00238, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| xor_digit-medium | SGD | No | No | ✗ | 74751.77s | 6.90e-01 | 100 | 0 | `ForeachSGD(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit-medium | AdamW | No | Yes | ✓ | 122686.49s | 9.86e-04 | 74 | 0 | `ForeachAdamW(lr=0.00712, betas=(0.458, 0.9735), shampoo_beta=0.009)` |
| xor_digit_rnn-medium | AdamW | No | No | ✗ | 109768.52s | 6.86e-01 | 100 | 0 | `ForeachAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | MARSAdamW | No | No | ✗ | 134477.92s | 6.86e-01 | 100 | 0 | `ForeachMARSAdamW(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | SGD | No | No | ✗ | 138793.06s | 6.89e-01 | 100 | 0 | `ForeachSGD(lr=0.00000, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_digit_rnn-medium | AdamW | No | Yes | ✗ | 158425.01s | inf | 100 | 0 | N/A |
| xor_sequence-medium | AdamW | No | No | ✓ | 18102.31s | 9.66e-03 | 6 | 0 | `ForeachAdamW(lr=0.00127, betas=(1.000, 1.0000), shampoo_beta=0.953)` |
| xor_sequence-medium | MARSAdamW | No | No | ✓ | 20712.87s | 9.41e-03 | 4 | 0 | `ForeachMARSAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_sequence-medium | SGD | No | No | ✗ | 8818.66s | 4.89e-01 | 100 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_sequence-medium | AdamW | No | Yes | ✓ | 34383.31s | 9.22e-03 | 22 | 0 | `ForeachAdamW(lr=0.00419, betas=(1.000, 1.0000), shampoo_beta=0.264)` |
| xor_sequence_rnn-medium | AdamW | No | No | ✓ | 94643.13s | 9.66e-03 | 11 | 0 | `ForeachAdamW(lr=0.00005, betas=(1.000, 1.0000), shampoo_beta=0.914)` |
| xor_sequence_rnn-medium | MARSAdamW | No | No | ✓ | 28024.41s | 7.37e-03 | 4 | 0 | `ForeachMARSAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_sequence_rnn-medium | SGD | No | No | ✓ | 2352.23s | 9.83e-03 | 4 | 0 | `ForeachSGD(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_sequence_rnn-medium | AdamW | No | Yes | ✓ | 31503.00s | 9.96e-03 | 6 | 0 | `ForeachAdamW(lr=0.00002, betas=(0.999, 1.0000), shampoo_beta=0.876)` |
| xor_spot-medium | AdamW | No | No | ✓ | 18761.73s | 6.33e-03 | 22 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.788, 1.0000), shampoo_beta=0.267)` |
| xor_spot-medium | MARSAdamW | No | No | ✓ | 59101.38s | 7.71e-03 | 30 | 0 | `ForeachMARSAdamW(lr=0.00111, betas=(0.441, 1.0000), shampoo_beta=1.000)` |
| xor_spot-medium | SGD | No | No | ✗ | 19029.57s | 6.92e-01 | 100 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot-medium | AdamW | No | Yes | ✓ | 16027.31s | 8.83e-03 | 12 | 0 | `ForeachAdamW(lr=0.00092, betas=(0.745, 1.0000), shampoo_beta=0.001)` |
| xor_spot_rnn-medium | AdamW | No | No | ✗ | 143729.13s | 6.89e-01 | 100 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-medium | MARSAdamW | No | No | ✗ | 159648.68s | inf | 100 | 0 | N/A |
| xor_spot_rnn-medium | SGD | No | No | ✗ | 27425.37s | 6.83e-01 | 100 | 0 | `ForeachSGD(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| xor_spot_rnn-medium | AdamW | No | Yes | ✗ | 158158.97s | inf | 100 | 0 | N/A |

## Errors


### gradient_delay-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### gradient_delay-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/gradient_delay.py", line 72, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 764, in estimate_condition_number_hvp
    hvp = compute_hvp(loss, params, v)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 739, in compute_hvp
    grads = torch.autograd.grad(loss, params, create_graph=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/__init__.py", line 496, in grad
    result = _engine_run_backward(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

```

### layer_wise_scale-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/layer_wise_scale.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### minimax-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 750, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 512])) must be the same as input size (torch.Size([]))

```

### minimax-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 750, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 512])) must be the same as input size (torch.Size([]))

```

### minimax-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 750, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 512])) must be the same as input size (torch.Size([]))

```

### minimax-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 579, in trial
    condition_number = estimate_condition_number_hvp(model, data, n_probes=20, n_samples=500)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 750, in estimate_condition_number_hvp
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)  # or appropriate loss
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/torch/nn/functional.py", line 3639, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([16, 512])) must be the same as input size (torch.Size([]))

```

### momentum_utilization-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/momentum_utilization.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### momentum_utilization-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/momentum_utilization.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### momentum_utilization-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/momentum_utilization.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### plateau_navigation-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/plateau_navigation.py", line 80, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### plateau_navigation-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/plateau_navigation.py", line 80, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### plateau_navigation-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/plateau_navigation.py", line 80, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### plateau_navigation-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/plateau_navigation.py", line 80, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_scale-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_scale.py", line 54, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_scale-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_scale.py", line 54, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_scale-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_scale.py", line 54, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_scale-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_scale.py", line 54, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_target-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_target.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_target-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_target.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_target-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_target.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### quadratic_varying_target-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/quadratic_varying_target.py", line 56, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### saddle_point-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/saddle_point.py", line 84, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### scale_invariant-medium - AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/scale_invariant.py", line 59, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### scale_invariant-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/scale_invariant.py", line 59, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### scale_invariant-medium - SGD
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/scale_invariant.py", line 59, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### scale_invariant-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/scale_invariant.py", line 59, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_digit_rnn-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_digit_rnn.py", line 69, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - MARSAdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```

### xor_spot_rnn-medium - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 118, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/xor_spot_rnn.py", line 87, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 704, in trial
    study.optimize(_optuna_objective, n_trials=trials)
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/home/tomjodrell/HeavyBall/venv/lib64/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 680, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 500, in objective
    raise Stop
benchmark.utils.Stop

```
