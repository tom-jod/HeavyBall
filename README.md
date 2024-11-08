# HeavyBall

A simple package of efficient optimizers

The goal is not to thrive for completeness, full maintenance or abstraction, but instead to provide a simple
largely static alternative to `torch.optim` with more and better optimizers.

## Getting started

```bash
pip install heavyball
```

```python
import torch
import heavyball

# Create a model
model = torch.nn.Linear(16, 1)

# Create an optimizer
optimizer = heavyball.PaLMForeachSFAdamW(model.parameters(), lr=1e-3)

x = torch.randn(128, 16)
y = torch.randn(128, 1)

for _ in range(1000):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
```

## Optimizers

| Name                                 | Description                                                                                                                                                       | Advantages / Disadvantages                                                                                                                                                                                                                                                                  |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ForeachAdamW**                     | More efficient (speed, memory) [AdamW](https://arxiv.org/abs/1711.05101)                                                                                          | + Faster than AdamW<br>+ Possibly more (numerically) stable                                                                                                                                                                                                                                 
| **ForeachLaProp**                    | More efficient (speed, memory) [LaProp](https://arxiv.org/abs/2002.04839)                                                                                         | + Same cost as AdamW<br>+ Marginally better converence (better proofs)<br>+ Higher hyperparameter stability<br>- Not a guaranteed win (can be neutral)<br>- No "Slingshot"                                                                                                                  |
| **ForeachADOPT**                     | More efficient (speed, memory) [ADOPT](https://arxiv.org/abs/2411.02853)                                                                                          | + Same cost as AdamW<br>+ Rigorous mathematical convergence proofs, even for challenging models (GANs)<br>- Empirically underperforms LaProp<br>- no bf16                                                                                                                                   |
| **ForeachSFAdamW**                   | More efficient (speed, memory) [ScheduleFree AdamW](https://arxiv.org/abs/2405.15682)                                                                             | + Same cost as AdamW, but better eval perf<br>+ Full control over hyperparameters                                                                                                                                                                                                           |
| **PaLMForeachSFAdamW**               | ForeachSFAdamW with [PaLM's beta2 schedule](https://arxiv.org/abs/2204.02311)                                                                                     | + Same cost as AdamW, but better eval perf<br>+ Less control, but faster early and more stable late convergence<br>+ ScheduleFree<br>- slow early convergence                                                                                                                               |
| **ForeachSOAP**                      | More efficient (speed, memory) [SOAP](https://arxiv.org/abs/2409.11321)                                                                                           | + Fastest convergence (loss-at-step)<br>+ Full control over hyperparameters<br>- more memory usage<br>- more hyperparameters<br>- higher overhead than AdamW (can be ammortized; better loss-at-second)                                                                                     |
| **PaLMForeachSOAP**                  | ForeachSOAP with [PaLM's beta2 schedule](https://arxiv.org/abs/2204.02311)                                                                                        | + Fastest convergence (loss-at-step)<br>+ Less control, but faster early and more stable late convergence<br>- more memory usage<br>- more hyperparameters<br>- higher overhead than AdamW (can be ammortized; better loss-at-second)                                                       |
| **SFPaLMForeachSOAP**                | ScheduleFree PaLMForeachSOAP                                                                                                                                      | + Fast convergence (loss-at-step)<br>+ less memory usage than PaLMForeachSOAP (more tham AdamW)<br>- slower initial convergence than PaLMForeachSOAP (but allows higher LRs)<br>- higher overhead than AdamW (can be ammortized)                                                            |
| **PrecondScheduleSFPaLMForeachSOAP** | SFPaLMForeachSOAP with [preconditioner schedule](https://github.com/lixilinx/psgd_torch/), matching the error of PrecondEvery=2 with the cost of PrecondEvery=512 | + Better initial convergence than SFPaLMForeachSOAP<br>+ Significantly faster (sec/it) later<br>+ less memory usage than PaLMForeachSOAP (more tham AdamW)<br>- slower initial convergence than PaLMForeachSOAP (but allows higher LRs)<br>- higher overhead than AdamW (can be ammortized) |
