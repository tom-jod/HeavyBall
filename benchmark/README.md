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

| Test | Description | Primary Challenge | In Suite |
|------|-------------|-------------------|:---------:|
| Adversarial Gradient | Tests optimizer robustness to misleading gradients | Gradient Direction & Stability | ✓ |
| Batch Size Scaling | Tests adaptation to batch size changes | Noise & Batch Effects | ✓ |
| Beale | Tests navigation of complex valley structure | Deep Valleys & Nonconvexity | ✓ |
| Char RNN | Tests sequential pattern learning | Long Sequences & Weak Gradients | |
| Discontinuous Gradient | Tests non-smooth landscape navigation | Discontinuities & Convergence | ✓ |
| Dynamic Landscape | Tests adaptation to shifting optima | Moving Targets & Changing Gradients | ✓ |
| Exploding Gradient | Tests handling of growing gradients | Gradient Explosion & Overflow | ✓ |
| Gradient Delay | Tests async update handling | Delayed Gradients & Async Updates | ✓ |
| Gradient Noise Scale | Tests noise level adaptation | Variable Noise & Scaling | ✓ |
| Grokking | Tests sudden learning after memorization | Phase Transitions & Memory | |
| Layer-wise Scale | Tests multi-layer gradient scaling | Layer Variation & Balance | ✓ |
| Loss Contour | Tests complex landscape navigation | Surface Complexity & Visualization | |
| Momentum Utilization | Tests momentum in oscillating landscapes | Oscillations & Momentum | ✓ |
| Noisy Matmul | Tests heavy noise robustness | Noise Robustness & Stability | ✓ |
| Plateau Navigation | Tests small gradient regions | Plateaus & Weak Gradients | ✓ |
| Powers | Tests polynomial optimization | Polynomial Surfaces & Curvature | ✓ |
| Powers Varying Target | Tests changing polynomial targets | Moving Targets & Surface Changes | ✓ |
| Quadratic Varying Scale | Tests dynamic parameter scaling | Scale Adaptation & Stability | ✓ |
| Quadratic Varying Target | Tests moving quadratic minima | Moving Targets & Quadratics | ✓ |
| Rastrigin | Tests multimodal landscape escape | Multiple Minima & Escape | ✓ |
| ReLU Boundaries | Tests ReLU decision boundary learning | Sharp Boundaries & Zero Gradients | |
| Rosenbrock | Tests curved valley navigation | Valley Navigation & Sharp Turns | ✓ |
| Saddle Point | Tests saddle point escape | Saddle Points & Escape | ✓ |
| Scale Invariant | Tests scale-invariant optimization | Scale Invariance & Robustness | ✓ |
| Sparse Gradient | Tests sparse update handling | Sparse Gradients & Embeddings | ✓ |
| XOR Digit | Tests RNN weak gradient learning (~RL) | Weak Gradients & Memory | ✓ |
| XOR Sequence | Tests RNN state compression (~AE, LLM) | Compression & Sequential | ✓ |
| XOR Spot | Tests RNN multiplicative effects (~RL) | Multiplicative & Memory | ✓ |
| Explicit Constraints | verifying convergence to the boundary of a constrained region | Explicit Constraints | ✓ |
