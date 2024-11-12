# HeavyBall - Benchmark

This test suite is meant to have fast execution, with each test case covering different optimization properties.

| Name         | Description                                                     
|--------------|-----------------------------------------------------------------|
| Beale        | Identify the correct valley                                     |
| Rosenbrock   | Descend through narrow, flat valley after initial steep descent |
| Quadratic    | Sanity check                                                    |
| Noisy Matmul | Test robustness to heavy noise                                  |
| XOR Sequence | Cram hidden state with good gradients (~AE, LLM)                |
| XOR Digit    | Simple RNN with weak gradients (~RL)                            |
| XOR Spot     | Multiply in RNN (~RL)                                           |

