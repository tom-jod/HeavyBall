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

| Test                     | Description                                        | Primary Challenge                   | No. Params| In Suite | Uses Minibatches | Difficulties | What does difficulty do?
|--------------------------|----------------------------------------------------|-------------------------------------|-----------|---|------
| Adversarial Gradient     | Tests optimizer robustness to misleading gradients | Gradient Direction & Stability      | 1024      | ✓ | ✗ | All | Decreases update frequency
| Batch Size Scaling       | Tests adaptation to batch size changes             | Noise & Batch Effects               | 1024      | ✓ | ✗ | All | Decreases max batch size
| Beale                    | Tests navigation of complex valley structure       | Deep Valleys & Nonconvexity         |   -       | ✓ | ✗ | Trivial | None
| Char RNN                 | Tests sequential pattern learning                  | Long Sequences & Weak Gradients     | 2,361,600 | ✗ | ✓ | None specified | None
| Constrained Optimization | Tests constrained optimization learning            | Balancing optimisation with validity|  16       | ✓ | ✗ | All | Larger penalty
| Discontinuous Gradient   | Tests non-smooth landscape navigation              | Discontinuities & Convergence       | 1024      | ✓ | ✗ | Trivial | None
| Dynamic Landscape        | Tests adaptation to shifting optima                | Moving Targets & Changing Gradients | 16,384    | ✓ | ✗ | All | Lower frequency
| Exploding Gradient       | Tests handling of growing gradients                | Gradient Explosion & Overflow       | 512       | ✓ | ✗ | All | Larger Scale (gradient explodes more rapidly)
| Gradient Delay           | Tests async update handling                        | Delayed Gradients & Async Updates   | 256       | ✓ | ✗ | All | Maximum delay
| Gradient Noise Scale     | Tests noise level adaptation                       | Variable Noise & Scaling            | 4096      | ✓ | ✗ | All | Decreases offset
| Grokking                 | Tests sudden learning after memorization           | Phase Transitions & Memory          | 22,017    | ✗ | ✓ | None specified | None
| Layer-wise Scale         | Tests multi-layer gradient scaling                 | Layer Variation & Balance           | 3072      | ✓ | ✗ | All | Scale increases (how many times larger the gradients from one layer are to another)
| Loss Contour             | Tests complex landscape navigation                 | Surface Complexity & Visualization  | 55,627    | ✗ | ✗ | None specified | None
| Minimax                  | Tests bilinear saddle point optimization           | Escaping the Saddle Point           |8 - 262,144| ✓ | ✓ | All | Increases number of parameters
| Momentum Utilization     | Tests momentum in oscillating landscapes           | Oscillations & Momentum             | 1024      | ✓ | ✗ | All | Increases weighting of oscillations in the quadratic term
| Noisy Matmul             | Tests heavy noise robustness                       | Noise Robustness & Stability        | 64        | ✓ | ✓ | All | The number of sequential operations
| Parameter scale          | Tests handling of multi-scale gradients            | Too Strong or Weak Gradients        | 3072      | ✓ | ✗ | Easy, Medium, Hard | Scales the size of initial parameter values
| Plateau Navigation       | Tests small gradient regions                       | Plateaus & Weak Gradients           |  -        | ✓ | ✗ | All | Sharpness of exit from the plateau
| Powers                   | Tests polynomial optimization     (easier version) | Polynomial Surfaces & Curvature     |256-32,768 | ✓ | ✗ | All | Size of the problem
| Powers Varying Target    | Tests changing polynomial targets (harder version) | Moving Targets & Surface Changes    |256-32,768 | ✓ | ✗ | All | Size of the problem
| Quadratic Varying Scale  | Tests dynamic parameter scaling                    | Scale Adaptation & Stability        |4 - 131,072| ✓ | ✗ | All | Size of the problem
| Quadratic Varying Target | Tests moving quadratic minima                      | Moving Targets & Quadratics         |4 - 131,072| ✓ | ✗ | All | Size of the problem
| Rastrigin                | Tests multimodal landscape escape                  | Multiple Minima & Escape            |   -       | ✓ | ✗ | Trivial | None
| ReLU Boundaries          | Tests ReLU decision boundary learning              | Sharp Boundaries & Zero Gradients   | 1,218     | ✗ | ✓ | None specified | None
| Rosenbrock               | Tests curved valley navigation                     | Valley Navigation & Sharp Turns     |   -       | ✓ | ✗ | Trivial | None
| Saddle Point             | Tests saddle point escape                          | Saddle Points & Escape              |   -       | ✓ | ✗ | All | None
| Scale Invariant          | Tests scale-invariant optimization                 | Scale Invariance & Robustness       | 512       | ✓ | ✗ | All | Increases range of values that parameters can take
| Sparse Gradient          | Tests sparse update handling                       | Sparse Gradients & Embeddings       | 65,536    | ✓ | ✗ | All | Fewer parameters get gradients
| Wide Linear              | Tests upper triangular matrix learning             | Discover sparse/structured solutions|16-268,435,456| ✓ | ✓ | All | Larger model
| XOR Digit (RNN)          | Tests RNN weak gradient learning using LSTM (RNN)  | Weak Gradients & Memory             | 8,641     | ✓ | ✓ | All | None
| XOR Sequence (RNN)       | Tests RNN state compression (~AE, LLM)             | Compression & Sequential            | 1,201     | ✓ | ✓ | All | None 
| XOR Spot (RNN)           | Tests RNN multiplicative effects (~RL)             | Multiplicative & Memory             | 8,769     | ✓ | ✓ | All | None
