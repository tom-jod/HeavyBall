# HeavyBall - Benchmark

### Getting Started
```BASH
# Install dependencies (`--use-deprecated=legacy-resolver` is required to install the latest BoTorch)
python3 -m pip install -r benchmark/requirements.txt --use-deprecated=legacy-resolver

# Small test run
python3 -m benchmark.beale --opt AdamW --steps 1000

# Reproduce the full benchmark
python3 -m benchmark.run_all_benchmarks --opt ForeachSOAP --opt LaProp --opt AdamW --opt Muon --opt ForeachCachedNewtonPSGD  --opt RMSprop --opt OrthoLaProp --opt ForeachSFAdamW --opt ForeachADOPT --opt LaPropOrtho --opt CachedPSGDKron --opt SignLaProp --opt ForeachSOLP --opt PSGDLRA --opt NewtonPSGDLRA --opt NewtonHybrid2PSGDKron --opt NewtonHybrid2PSGDLRA --opt mars-NewtonHybrid2PSGDLRA --opt MSAMLaProp --opt mars-adaptive-NewtonHybrid2PSGDKron  --opt mars-ortho-NewtonHybrid2PSGDKron --opt MuonLaProp --opt mars-unscaled-NewtonHybrid2PSGDKron --opt mars-NewtonHybrid2PSGDKron --opt cautious-AdamW --opt unscaled_cautious-AdamW --opt mars-AdamW  --dtype float32 --steps 1000000 --trials 1000 --parallelism 256 --seeds 1 --difficulties trivial --difficulties easy --difficulties medium --difficulties hard --difficulties extreme --difficulties nightmare --timeout 2880
```

### Benchmark Tests

| Test                     | Description                                        | Primary Challenge                   | In Suite | Uses Minibatches | Difficulties |
|--------------------------|----------------------------------------------------|-------------------------------------|---|---|------
| Adversarial Gradient     | Tests optimizer robustness to misleading gradients | Gradient Direction & Stability      | ✓ | ✗ | All
| Batch Size Scaling       | Tests adaptation to batch size changes             | Noise & Batch Effects               | ✓ | ✗ | All
| Beale                    | Tests navigation of complex valley structure       | Deep Valleys & Nonconvexity         | ✓ | ✗ | Trivial
| Char RNN                 | Tests sequential pattern learning                  | Long Sequences & Weak Gradients     | ✗ | ✓ | None specified
| Constrained Optimization | Tests constrained optimization learning            | Balancing optimisation with validity| ✓ | ✗ | All
| Discontinuous Gradient   | Tests non-smooth landscape navigation              | Discontinuities & Convergence       | ✓ | ✗ | Trivial
| Dynamic Landscape        | Tests adaptation to shifting optima                | Moving Targets & Changing Gradients | ✓ | ✗ | All
| Exploding Gradient       | Tests handling of growing gradients                | Gradient Explosion & Overflow       | ✓ | ✗ | All
| Gradient Delay           | Tests async update handling                        | Delayed Gradients & Async Updates   | ✓ | ✗ | All
| Gradient Noise Scale     | Tests noise level adaptation                       | Variable Noise & Scaling            | ✓ | ✗ | All
| Grokking                 | Tests sudden learning after memorization           | Phase Transitions & Memory          | ✗ | ✓ | None specified
| Layer-wise Scale         | Tests multi-layer gradient scaling                 | Layer Variation & Balance           | ✓ | ✗ | All
| Loss Contour             | Tests complex landscape navigation                 | Surface Complexity & Visualization  | ✗ | ✗ | None specified
| Minimax                  | Tests bilinear saddle point optimization           | Escaping the Saddle Point           | ✓ | ✓ | All
| Momentum Utilization     | Tests momentum in oscillating landscapes           | Oscillations & Momentum             | ✓ | ✗ | All
| Noisy Matmul             | Tests heavy noise robustness                       | Noise Robustness & Stability        | ✓ | ✓ | All
| Parameter scale          | Tests handling of multi-scale gradients            | Too Strong or Weak Gradients        | ✓ | ✗ | Easy, Medium, Hard
| Plateau Navigation       | Tests small gradient regions                       | Plateaus & Weak Gradients           | ✓ | ✗ | All
| Powers                   | Tests polynomial optimization                      | Polynomial Surfaces & Curvature     | ✓ | ✗ | All
| Powers Varying Target    | Tests changing polynomial targets                  | Moving Targets & Surface Changes    | ✓ | ✗ | All
| Quadratic Varying Scale  | Tests dynamic parameter scaling                    | Scale Adaptation & Stability        | ✓ | ✗ | All
| Quadratic Varying Target | Tests moving quadratic minima                      | Moving Targets & Quadratics         | ✓ | ✗ | All
| Rastrigin                | Tests multimodal landscape escape                  | Multiple Minima & Escape            | ✓ | ✗ | Trivial
| ReLU Boundaries          | Tests ReLU decision boundary learning              | Sharp Boundaries & Zero Gradients   | ✗ | ✓ | None specified
| Rosenbrock               | Tests curved valley navigation                     | Valley Navigation & Sharp Turns     | ✓ | ✗ | Trivial
| Saddle Point             | Tests saddle point escape                          | Saddle Points & Escape              | ✓ | ✗ | All
| Scale Invariant          | Tests scale-invariant optimization                 | Scale Invariance & Robustness       | ✓ | ✗ | All
| Sparse Gradient          | Tests sparse update handling                       | Sparse Gradients & Embeddings       | ✓ | ✗ | All
| Wide Linear              | Tests upper triangular matrix learning             | Discover sparse/structured solutions| ✓ | ✓ | All
| XOR Digit (RNN)          | Tests RNN weak gradient learning using LSTM (RNN)  | Weak Gradients & Memory             | ✓ | ✓ | All
| XOR Sequence (RNN)       | Tests RNN state compression (~AE, LLM)             | Compression & Sequential            | ✓ | ✓ | All
| XOR Spot (RNN)           | Tests RNN multiplicative effects (~RL)             | Multiplicative & Memory             | ✓ | ✓ | All
