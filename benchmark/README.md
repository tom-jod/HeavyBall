# HeavyBall - Benchmark

This test suite is meant to have fast execution, with each test case covering different optimization properties.

| Name         | Description                                                     |
|--------------|-----------------------------------------------------------------|
| Beale        | Identify the correct valley                                      |
| Rosenbrock   | Descend through narrow, flat valley after initial steep descent |
| Quadratic    | Sanity check                                                    |
| Noisy Matmul | Test robustness to heavy noise                                  |
| XOR Sequence | Cram hidden state with good gradients (~AE, LLM)                |
| XOR Digit    | Simple RNN with weak gradients (~RL)                            |
| XOR Spot     | Multiply in RNN (~RL)                                           |
| Saddle Point | Test ability to escape saddle points                            |
| Discontinuous Gradient | Test robustness to non-smooth landscapes              |
| Plateau Navigation | Test handling of regions with very small gradients        |
| Scale Invariant | Test handling of different parameter scales                  |
| Momentum Utilization | Test effective use of momentum in oscillating landscapes |
| Batch Size Scaling | Test adaptation to different batch sizes and noise scales |
| Sparse Gradient | Test handling of sparse updates (embeddings, attention)      |
| Layer-wise Scale | Test handling of different gradient scales across layers    |
| Gradient Delay | Test handling of asynchronous/delayed updates                 |
| Ill-Conditioned | Test optimization of poorly-conditioned problems             |
| Gradient Noise Scale | Test adaptation to changing noise levels                |
| Memory Length | Test effective use of optimization history                     |
| Adversarial Gradient | Test robustness to misleading gradient patterns        |
