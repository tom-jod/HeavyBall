# Benchmark Results
Generated: 2025-05-21 10:44:23.002081
Last updated: 2025-05-21 10:44:23.002090

## Summary (In Progress)

| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |
|-----------|---|---|---------|----------|------|
| AdamW | No | Yes | 23/28 | 15742.80s | 16.6 |
| AdamWScheduled | No | Yes | 24/28 | 13787.73s | 17.0 |

## Details

| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |
|-----------|-----------|---------|---|---|----------|------|---|---|----------------|
| adversarial_gradient-trivial | AdamW | No | Yes | ✗ | 19254.99s | -4.58e-02 | 1000 | 0 | `ForeachAdamW(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| adversarial_gradient-trivial | AdamWScheduled | No | Yes | ✗ | 18041.65s | -4.58e-02 | 1000 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| batch_size_scaling-trivial | AdamW | No | Yes | ✓ | 1761.26s | 1.05e-16 | 6 | 0 | `ForeachAdamW(lr=0.03357, betas=(0.872, 1.0000), shampoo_beta=0.388)` |
| batch_size_scaling-trivial | AdamWScheduled | No | Yes | ✓ | 3307.46s | 1.11e-16 | 7 | 0 | `ForeachAdamWScheduled(lr=0.00151, betas=(0.988, 1.0000), shampoo_beta=0.994)` |
| beale-trivial | AdamW | No | Yes | ✓ | 46416.79s | inf | 33 | 0 | `ForeachAdamW(lr=0.01008, betas=(0.997, 0.9998), shampoo_beta=0.507)` |
| beale-trivial | AdamWScheduled | No | Yes | ✓ | 18273.97s | inf | 38 | 0 | `ForeachAdamWScheduled(lr=0.00465, betas=(0.898, 0.9933), shampoo_beta=0.487)` |
| constrained_optimization-trivial | AdamW | No | Yes | ✓ | 11062.62s | 1.00e+00 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| constrained_optimization-trivial | AdamWScheduled | No | Yes | ✓ | 11143.96s | 1.00e+00 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| discontinuous_gradient-trivial | AdamW | No | Yes | ✓ | 29223.26s | 1.00e-08 | 109 | 0 | `ForeachAdamW(lr=0.00135, betas=(0.000, 1.0000), shampoo_beta=0.000)` |
| discontinuous_gradient-trivial | AdamWScheduled | No | Yes | ✓ | 34793.43s | 1.00e-08 | 109 | 0 | `ForeachAdamWScheduled(lr=0.00092, betas=(0.000, 1.0000), shampoo_beta=0.000)` |
| dynamic_landscape-trivial | AdamW | No | Yes | ✓ | 2390.00s | 9.85e-03 | 8 | 0 | `ForeachAdamW(lr=17.56343, betas=(0.985, 1.0000), shampoo_beta=0.981)` |
| dynamic_landscape-trivial | AdamWScheduled | No | Yes | ✓ | 1063.41s | 9.87e-03 | 6 | 0 | `ForeachAdamWScheduled(lr=1.89986, betas=(0.946, 0.9997), shampoo_beta=0.903)` |
| gradient_delay-trivial | AdamW | No | Yes | ✓ | 1033.35s | 9.90e-05 | 9 | 0 | `ForeachAdamW(lr=0.01047, betas=(0.960, 1.0000), shampoo_beta=0.959)` |
| gradient_delay-trivial | AdamWScheduled | No | Yes | ✓ | 1530.14s | 9.95e-05 | 12 | 0 | `ForeachAdamWScheduled(lr=0.00240, betas=(0.982, 1.0000), shampoo_beta=0.396)` |
| gradient_noise_scale-trivial | AdamW | No | Yes | ✓ | 18080.77s | 1.00e-06 | 35 | 0 | `ForeachAdamW(lr=0.00073, betas=(1.000, 1.0000), shampoo_beta=1.000)` |
| gradient_noise_scale-trivial | AdamWScheduled | No | Yes | ✓ | 14512.80s | 1.14e-06 | 15 | 0 | `ForeachAdamWScheduled(lr=0.00338, betas=(0.993, 1.0000), shampoo_beta=0.480)` |
| layer_wise_scale-trivial | AdamW | No | Yes | ✓ | 20049.09s | 9.89e-05 | 26 | 0 | `ForeachAdamW(lr=0.88257, betas=(0.914, 0.9765), shampoo_beta=0.999)` |
| layer_wise_scale-trivial | AdamWScheduled | No | Yes | ✓ | 15800.60s | 1.00e-04 | 19 | 0 | `ForeachAdamWScheduled(lr=0.00067, betas=(1.000, 1.0000), shampoo_beta=0.521)` |
| minimax-trivial | AdamW | No | Yes | ✗ | 26269.08s | inf | 1000 | 0 | N/A |
| minimax-trivial | AdamWScheduled | No | Yes | ✗ | 52699.69s | 1.55e+00 | 1000 | 0 | `ForeachAdamWScheduled(lr=0.00100, betas=(0.100, 0.0010), shampoo_beta=0.001)` |
| momentum_utilization-trivial | AdamW | No | Yes | ✓ | 13040.66s | -2.83e-05 | 22 | 0 | `ForeachAdamW(lr=0.59148, betas=(0.906, 1.0000), shampoo_beta=0.997)` |
| momentum_utilization-trivial | AdamWScheduled | No | Yes | ✓ | 14310.51s | -9.17e-05 | 22 | 0 | `ForeachAdamWScheduled(lr=0.73653, betas=(0.845, 0.5996), shampoo_beta=0.999)` |
| noisy_matmul-trivial | AdamW | No | Yes | ✗ | 55296.99s | inf | 1000 | 0 | N/A |
| noisy_matmul-trivial | AdamWScheduled | No | Yes | ✓ | 8251.21s | 1.04e-14 | 15 | 0 | `ForeachAdamWScheduled(lr=0.00182, betas=(0.924, 0.9954), shampoo_beta=0.984)` |
| parameter_scale-trivial | AdamW | No | Yes | ✓ | 34357.04s | 9.59e-05 | 16 | 0 | `ForeachAdamW(lr=3.43573, betas=(0.994, 0.9687), shampoo_beta=1.000)` |
| parameter_scale-trivial | AdamWScheduled | No | Yes | ✓ | 36726.55s | 9.68e-05 | 61 | 0 | `ForeachAdamWScheduled(lr=1.63032, betas=(0.972, 0.9955), shampoo_beta=1.000)` |
| plateau_navigation-trivial | AdamW | No | Yes | ✓ | 10349.10s | 9.96e-05 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| plateau_navigation-trivial | AdamWScheduled | No | Yes | ✓ | 9837.47s | 9.96e-05 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| quadratic_varying_scale-trivial | AdamW | No | Yes | ✓ | 44389.27s | 2.14e-14 | 45 | 0 | `ForeachAdamW(lr=0.00171, betas=(0.936, 0.9911), shampoo_beta=1.000)` |
| quadratic_varying_scale-trivial | AdamWScheduled | No | Yes | ✓ | 18221.74s | 6.52e-14 | 23 | 0 | `ForeachAdamWScheduled(lr=3.80246, betas=(0.981, 0.9995), shampoo_beta=1.000)` |
| quadratic_varying_target-trivial | AdamW | No | Yes | ✓ | 24430.00s | 2.67e-15 | 27 | 0 | `ForeachAdamW(lr=0.98099, betas=(0.967, 0.9986), shampoo_beta=0.103)` |
| quadratic_varying_target-trivial | AdamWScheduled | No | Yes | ✓ | 35015.02s | 9.25e-16 | 32 | 0 | `ForeachAdamWScheduled(lr=0.27884, betas=(0.924, 0.9968), shampoo_beta=0.989)` |
| rastrigin-trivial | AdamW | No | Yes | ✗ | 55200.81s | inf | 1000 | 0 | N/A |
| rastrigin-trivial | AdamWScheduled | No | Yes | ✗ | 55341.84s | inf | 1000 | 0 | N/A |
| rosenbrock-trivial | AdamW | No | Yes | ✓ | 15500.42s | 9.96e-10 | 9 | 0 | `ForeachAdamW(lr=0.00465, betas=(0.950, 0.9995), shampoo_beta=0.999)` |
| rosenbrock-trivial | AdamWScheduled | No | Yes | ✓ | 14851.33s | 9.99e-10 | 6 | 0 | `ForeachAdamWScheduled(lr=0.03823, betas=(0.835, 1.0000), shampoo_beta=0.865)` |
| saddle_point-trivial | AdamW | No | Yes | ✓ | 12457.05s | 9.98e-02 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| saddle_point-trivial | AdamWScheduled | No | Yes | ✓ | 12654.66s | 9.98e-02 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| scale_invariant-trivial | AdamW | No | Yes | ✓ | 14670.08s | 9.94e-04 | 6 | 0 | `ForeachAdamW(lr=0.03245, betas=(0.997, 1.0000), shampoo_beta=0.292)` |
| scale_invariant-trivial | AdamWScheduled | No | Yes | ✓ | 19127.53s | 9.93e-04 | 14 | 0 | `ForeachAdamWScheduled(lr=2.49202, betas=(0.977, 0.9694), shampoo_beta=0.994)` |
| sparse_gradient-trivial | AdamW | No | Yes | ✓ | 464.57s | 9.88e-05 | 8 | 0 | `ForeachAdamW(lr=0.00361, betas=(0.989, 1.0000), shampoo_beta=0.997)` |
| sparse_gradient-trivial | AdamWScheduled | No | Yes | ✓ | 176.99s | 9.91e-05 | 6 | 0 | `ForeachAdamWScheduled(lr=0.02586, betas=(0.965, 1.0000), shampoo_beta=0.559)` |
| wide_linear-trivial | AdamW | No | Yes | ✗ | 55041.20s | inf | 1000 | 0 | N/A |
| wide_linear-trivial | AdamWScheduled | No | Yes | ✗ | 53282.15s | inf | 1000 | 0 | N/A |
| xor_digit-trivial | AdamW | No | Yes | ✓ | 2513.42s | 9.82e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit-trivial | AdamWScheduled | No | Yes | ✓ | 2500.27s | 9.80e-04 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamW | No | Yes | ✓ | 15506.60s | 9.91e-04 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_digit_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 14966.94s | 9.88e-04 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamW | No | Yes | ✓ | 18992.27s | 9.34e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence-trivial | AdamWScheduled | No | Yes | ✓ | 18218.99s | 9.34e-03 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamW | No | Yes | ✓ | 18919.59s | 9.94e-03 | 2 | 0 | `ForeachAdamW(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_sequence_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 19086.33s | 9.94e-03 | 2 | 0 | `ForeachAdamWScheduled(lr=0.00004, betas=(1.000, 1.0000), shampoo_beta=0.972)` |
| xor_spot-trivial | AdamW | No | Yes | ✓ | 521.29s | 9.29e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot-trivial | AdamWScheduled | No | Yes | ✓ | 557.18s | 9.76e-03 | 4 | 0 | `ForeachAdamWScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamW | No | Yes | ✓ | 5955.94s | 9.87e-03 | 4 | 0 | `ForeachAdamW(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |
| xor_spot_rnn-trivial | AdamWScheduled | No | Yes | ✓ | 5976.93s | 9.65e-03 | 4 | 0 | `ForeachAdamWScheduled(lr=0.03492, betas=(1.000, 1.0000), shampoo_beta=0.408)` |

## Errors


### minimax-trivial - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/minimax.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 415, in objective
    raise Stop
benchmark.utils.Stop

```

### noisy_matmul-trivial - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/noisy_matmul.py", line 62, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 417, in objective
    raise Stop
benchmark.utils.Stop

```

### rastrigin-trivial - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/rastrigin.py", line 86, in main
    model = trial(
            ^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 417, in objective
    raise Stop
benchmark.utils.Stop

```

### rastrigin-trivial - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/rastrigin.py", line 86, in main
    model = trial(
            ^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 417, in objective
    raise Stop
benchmark.utils.Stop

```

### wide_linear-trivial - mars-AdamW
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/wide_linear.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 417, in objective
    raise Stop
benchmark.utils.Stop

```

### wide_linear-trivial - mars-AdamWScheduled
```
Traceback (most recent call last):
  File "/home/tomjodrell/HeavyBall/benchmark/run_all_benchmarks.py", line 117, in run_benchmark
    module.main(**arguments)
  File "/home/tomjodrell/HeavyBall/benchmark/wide_linear.py", line 57, in main
    trial(
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 612, in trial
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
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 588, in _optuna_objective
    out = obj.objective((lr, one_minus_beta1, one_minus_beta2, one_minus_shampoo_beta))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tomjodrell/HeavyBall/benchmark/utils.py", line 415, in objective
    raise Stop
benchmark.utils.Stop

```
