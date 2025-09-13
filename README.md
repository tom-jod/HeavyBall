
## Quickstart

**Install:**
```bash
pip install heavyball
```

**Basic usage:**
```python
import torch
from torch import nn
from heavyball import ForeachAdamW

model = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)
optimizer = ForeachAdamW(model.parameters(), lr=1e-3)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
```

## Benchmarks

> Reproduce toy problem benchmarks with:
> ```bash
> python3 -m benchmark.run_all_benchmarks --opt ForeachSOAP --opt LaProp --opt AdamW --opt Muon --opt ForeachCachedNewtonPSGD  --opt RMSprop --opt OrthoLaProp --opt ForeachSFAdamW --opt ForeachADOPT --opt LaPropOrtho --opt CachedPSGDKron --opt SignLaProp --opt ForeachSOLP --opt PSGDLRA --opt NewtonPSGDLRA --opt NewtonHybrid2PSGDKron --opt NewtonHybrid2PSGDLRA --opt mars-NewtonHybrid2PSGDLRA --opt MSAMLaProp --opt mars-adaptive-NewtonHybrid2PSGDKron  --opt mars-ortho-NewtonHybrid2PSGDKron --opt MuonLaProp --opt mars-unscaled-NewtonHybrid2PSGDKron --opt mars-NewtonHybrid2PSGDKron --opt cautious-AdamW --opt unscaled_cautious-AdamW --opt mars-AdamW  --dtype float32 --steps 1000000 --trials 1000 --parallelism 256 --seeds 1 --difficulties trivial --difficulties easy --difficulties medium --difficulties hard --difficulties extreme --difficulties nightmare --timeout 2880
> ```
>  Or run just a single toy problem:
>  ```bash
> python3 -m benchmark.beale --opt AdamW --steps 27000 --trials 20
> ```
> Reproduce real world problem benchmarks (MNIST, SVHN, Tolstoi_RNN, CIFAR10-wide, CIFAR100) with:
>  ```bash
> python3 -m benchmark.MNIST --opt AdamW --steps 27000 --trials 20
> ```
>
> To efficiently run multiple repeats of an experiment over multiple optimizers use:
> ```bash
> python3 benchmark/benchmark_runner.py MNIST.py "AdamW, SFAdamW" --runs-per-optimizer=3 --runtime-limit=99999 --trials=20 --step-hint=27000 --steps=27000
> ```
> Alternatively to use time based stopping rather than step based stopping, set steps to 0 and specify the runtime and step hint:
> ```bash
> python3 benchmark/benchmark_runner.py MNIST.py "AdamW, SFAdamW" --runs-per-optimizer=3 --runtime-limit=472 --trials=20 --step-hint=27000 --steps=0
> ```
> 


## License

BSD 3-Clause — see the [LICENSE](LICENSE) file.

---


[pypi]: https://pypi.org/project/heavyball/
[license]: LICENSE
[tracker]: https://github.com/HomebrewML/HeavyBall/issues
