from __future__ import annotations

import functools
import math
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy
import numpy as np
import optuna
import optunahub
import pandas as pd
import torch
from botorch.utils.sampling import manual_seed
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna_integration.botorch import (
    ehvi_candidates_func,
    logei_candidates_func,
    qehvi_candidates_func,
    qei_candidates_func,
    qparego_candidates_func,
)
from torch import Tensor
from torch.nn import functional as F

from heavyball.utils import scalar_guard

_MAXINT32 = (1 << 31) - 1
_SAMPLER_KEY = "auto:sampler"


class SimpleAPIBaseSampler(BaseSampler):
    def __init__(
        self,
        search_space: dict[str, BaseDistribution] = None,
    ):
        self.search_space = search_space

    def suggest_all(self, trial: FrozenTrial):
        return {k: trial._suggest(k, dist) for k, dist in self.search_space.items()}


def _get_default_candidates_func(
    n_objectives: int,
    has_constraint: bool,
    consider_running_trials: bool,
) -> Callable[
    [
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        Tensor | None,
    ],
    Tensor,
]:
    """
    The original is available at https://github.com/optuna/optuna-integration/blob/156a8bc081322791015d2beefff9373ed7b24047/optuna_integration/botorch/botorch.py under the MIT License
    """
    if n_objectives > 3 and not has_constraint and not consider_running_trials:
        return ehvi_candidates_func
    elif n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    elif consider_running_trials:
        return qei_candidates_func
    else:
        return logei_candidates_func


@functools.lru_cache(maxsize=None)
def bound_to_torch(bound: bytes, shape: tuple, device: str):
    bound = np.frombuffer(bound, dtype=np.float64).reshape(shape)
    bound = np.transpose(bound, (1, 0))
    return torch.from_numpy(bound).to(torch.device(device))


@functools.lru_cache(maxsize=None)
def nextafter(x: Union[float, int], y: Union[float, int]) -> Union[float, int]:
    return numpy.nextafter(x, y)


def _untransform_numerical_param_torch(
    trans_param: Union[float, int, Tensor],
    distribution: BaseDistribution,
    transform_log: bool,
) -> Tensor:
    d = distribution

    if isinstance(d, FloatDistribution):
        if d.log:
            param = trans_param.exp() if transform_log else trans_param
            if d.single():
                return param
            return param.clamp(max=nextafter(d.high, d.high - 1))

        if d.step is not None:
            scaled = ((trans_param - d.low) / d.step).round() * d.step + d.low
            return scaled.clamp(min=d.low, max=d.high)

        if d.single():
            return trans_param

        return trans_param.clamp(max=nextafter(d.high, d.high - 1))

    if not isinstance(d, IntDistribution):
        raise ValueError(f"Unexpected distribution type: {type(d)}")

    if d.log:
        param = trans_param.exp().round() if transform_log else trans_param
    else:
        param = ((trans_param - d.low) / d.step).round() * d.step + d.low
    param = param.clamp(min=d.low, max=d.high)
    return param.to(torch.int64)


@torch.no_grad()
def untransform(self: _SearchSpaceTransform, trans_params: Tensor) -> dict[str, Any]:
    assert trans_params.shape == (self._raw_bounds.shape[0],)

    if self._transform_0_1:
        trans_params = self._raw_bounds[:, 0] + trans_params * (self._raw_bounds[:, 1] - self._raw_bounds[:, 0])

    params = {}

    for (name, distribution), encoded_columns in zip(self._search_space.items(), self.column_to_encoded_columns):
        if isinstance(distribution, CategoricalDistribution):
            raise ValueError("We don't support categorical parameters.")
        else:
            param = _untransform_numerical_param_torch(trans_params[encoded_columns], distribution, self._transform_log)

        params[name] = param

    return {n: v.item() for n, v in params.items()}


class BoTorchSampler(SimpleAPIBaseSampler):
    """
    A significantly more efficient implementation of `BoTorchSampler` from Optuna - keeps more on the GPU / in torch
    The original is available at https://github.com/optuna/optuna-integration/blob/156a8bc081322791015d2beefff9373ed7b24047/optuna_integration/botorch/botorch.py under the MIT License
    The original API is kept for backward compatibility, but many arguments are ignored to improve maintainability.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] = None,
        *,
        candidates_func: None = None,
        constraints_func: None = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        trial_chunks: int = 128,
    ):
        assert constraints_func is None
        assert candidates_func is None
        assert consider_running_trials is False
        assert independent_sampler is None
        self._candidates_func = None
        self._independent_sampler = RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self.trial_chunks = trial_chunks

        self._study_id: int | None = None
        self.search_space = search_space
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device or torch.device("cpu")
        self.seen_trials = set()
        self._values = None
        self._params = None
        self._index = 0

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self.search_space

    @torch.no_grad()
    def _preprocess_trials(
        self, trans: _SearchSpaceTransform, study: Study, trials: list[FrozenTrial]
    ) -> Tuple[int, Tensor, Tensor]:
        new_trials = []
        for trial in trials:
            tid: int = trial._trial_id
            if tid not in self.seen_trials:
                self.seen_trials.add(tid)
                new_trials.append(trial)
        trials = new_trials

        n_objectives = len(study.directions)
        if not new_trials:
            return n_objectives, self._values[: self._index], self._params[: self._index]

        n_completed_trials = len(trials)
        values: numpy.ndarray = numpy.empty((n_completed_trials, n_objectives), dtype=numpy.float64)
        params: numpy.ndarray = numpy.empty((n_completed_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        for trial_idx, trial in enumerate(trials):
            if trial.state != TrialState.COMPLETE:
                raise ValueError(f"TrialState must be COMPLETE, but {trial.state} was found.")

            params[trial_idx] = trans.transform(trial.params)
            values[trial_idx, :] = np.array(trial.values)

        for obj_idx, direction in enumerate(study.directions):
            if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                values[:, obj_idx] *= -1

        if self._values is None:
            self._values = torch.zeros((self.trial_chunks, n_objectives), dtype=torch.float64, device=self._device)
            self._params = torch.zeros(
                (self.trial_chunks, trans.bounds.shape[0]), dtype=torch.float64, device=self._device
            )
        spillage = (self._index + n_completed_trials) - self._values.size(0)
        if spillage > 0:
            pad = int(math.ceil(spillage / self.trial_chunks) * self.trial_chunks)
            self._values = F.pad(self._values, (0, 0, 0, pad))
            self._params = F.pad(self._params, (0, 0, 0, pad))
        self._values[self._index : self._index + n_completed_trials] = torch.from_numpy(values)
        self._params[self._index : self._index + n_completed_trials] = torch.from_numpy(params)
        self._index += n_completed_trials

        return n_objectives, self._values[: self._index], self._params[: self._index]

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        assert isinstance(search_space, dict)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        n_completed_trials = len(completed_trials)
        if n_completed_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives, values, params = self._preprocess_trials(trans, study, completed_trials)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(
                n_objectives=n_objectives, has_constraint=False, consider_running_trials=False
            )

        bounds = bound_to_torch(trans.bounds.tobytes(), trans.bounds.shape, str(self._device))

        with manual_seed(self._seed):
            candidates = self._candidates_func(params, values, None, bounds, None)
            if self._seed is not None:
                self._seed += 1

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )
        return untransform(trans, candidates)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(numpy.iinfo(numpy.int32).max)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


def _convert_to_hebo_design_space(search_space: dict[str, BaseDistribution]) -> DesignSpace:
    if not search_space:
        raise ValueError("Empty search space.")
    design_space = []
    for name, distribution in search_space.items():
        config: dict[str, Any] = {"name": name}
        if isinstance(distribution, (FloatDistribution, IntDistribution)):
            if not distribution.log and distribution.step is not None:
                config["type"] = "int"
                n_steps = int(np.round((distribution.high - distribution.low) / distribution.step + 1))
                config["lb"] = 0
                config["ub"] = n_steps - 1
            else:
                config["lb"] = distribution.low
                config["ub"] = distribution.high
                if distribution.log:
                    config["type"] = "pow_int" if isinstance(distribution, IntDistribution) else "pow"
                else:
                    assert not isinstance(distribution, IntDistribution)
                    config["type"] = "num"
        else:
            raise NotImplementedError(f"Unsupported distribution: {distribution}")

        design_space.append(config)
    return DesignSpace().parse(design_space)


class HEBOSampler(optunahub.samplers.SimpleBaseSampler, SimpleAPIBaseSampler):
    """
    Simplified version of https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/hebo/sampler.py
    modified under the MIT License
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        *,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        assert constant_liar is False
        assert independent_sampler is None
        self._hebo = HEBO(_convert_to_hebo_design_space(search_space), scramble_seed=self._seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._rng = np.random.default_rng(seed)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params = {}
        for name, row in self._hebo.suggest().items():
            if name not in search_space:
                continue

            dist = search_space[name]
            if isinstance(dist, (IntDistribution, FloatDistribution)) and not dist.log and dist.step is not None:
                step_index = row.iloc[0]
                params[name] = dist.low + step_index * dist.step
            else:
                params[name] = row.iloc[0]
        return params

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._hebo is None or values is None:
            return
        sign = 1 if study.direction == StudyDirection.MINIMIZE else -1
        values = np.array([values[0]])
        worst_value = np.nanmax(values) if study.direction == StudyDirection.MINIMIZE else np.nanmin(values)
        nan_padded_values = sign * np.where(np.isnan(values), worst_value, values)[:, np.newaxis]
        params = pd.DataFrame([trial.params])
        for name, dist in trial.distributions.items():
            if isinstance(dist, (IntDistribution, FloatDistribution)) and not dist.log and dist.step is not None:
                params[name] = (params[name] - dist.low) / dist.step

        self._hebo.observe(params, nan_padded_values)

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self.search_space

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)


class FastINGO:
    """
    Taken from https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/implicit_natural_gradient/sampler.py
    under the MIT License
    """

    def __init__(
        self,
        mean: np.ndarray,
        inv_sigma: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        last_n: int = 4096,
        loco_step_size: float = 1,
        device="cuda",
        batchnorm_decay: float = 0.99,
        score_decay: float = 0.99,
    ) -> None:
        n_dimension = len(mean)
        if population_size is None:
            population_size = 4 + int(np.floor(3 * np.log(n_dimension)))
            population_size = 2 * (population_size // 2)

        self.last_n = last_n
        self.batchnorm_decay = batchnorm_decay
        self.score_decay = score_decay
        self._learning_rate = learning_rate or 1.0 / np.sqrt(n_dimension)
        self._mean = torch.from_numpy(mean).to(device)
        self._sigma = torch.from_numpy(inv_sigma).to(device)
        self._lower = torch.from_numpy(lower).to(device)
        self._upper = torch.from_numpy(upper).to(device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0x123123 if seed is None else seed)
        self.loco_step_size = loco_step_size
        self._population_size = population_size
        self.device = device

        self._ys = None
        self._means = None
        self._z = None
        self._stds = None
        self._g = 0

    @torch.no_grad()
    def _concat(self, name, x):
        item = getattr(self, name, None)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        elif not isinstance(x, torch.Tensor):
            x = scalar_guard(x, self._mean).view(1)
        if item is not None:
            x = torch.cat((item, x), dim=0)[-self.last_n :]
        setattr(self, name, x)

    @property
    def dim(self) -> int:
        return self._mean.shape[0]

    @property
    def generation(self) -> int:
        return self._g

    @property
    def population_size(self) -> int:
        return self._population_size

    @torch.no_grad()
    def ask(self) -> np.ndarray:
        dimension = self._mean.shape[0]
        z = torch.randn(dimension, generator=self.generator, device=self.device, dtype=torch.float64)
        self._concat("_z", z[None])
        self._concat("_means", self._mean[None])
        self._concat("_stds", self._sigma[None])
        x = z / self._sigma.clamp(min=1e-8).sqrt() + self._mean
        return x.clamp(min=self._lower, max=self._upper).cpu().numpy()

    @torch.no_grad()
    def tell(self, y: float) -> None:
        self._g += 1
        self._concat("_ys", y)
        y = self._ys
        if y.numel() <= 2:
            return

        y = y + torch.where(y.min() <= 0, 1e-8 - y.min(), 0)
        y = y.log()

        ema = -torch.arange(y.size(0), device=y.device, dtype=y.dtype)
        weight = self.batchnorm_decay**ema
        weight = weight / weight.sum().clamp(min=1e-8)
        y_mean = weight @ y
        y_mean_sq = weight @ y.square()
        y_std = (y_mean_sq - y_mean.square()).clamp(min=1e-8).sqrt()
        score = (y.view(-1, 1) - y_mean) / y_std

        z = self._z
        mean_orig = self._means
        sigma_orig = self._stds
        mean_grad = score * (z / sigma_orig.clamp(min=1e-8).sqrt())
        sigma_grad = -score * z.square() * sigma_orig
        target_mean = mean_orig - mean_grad * self.loco_step_size  # MSE(current, target)
        target_sigma = sigma_orig - sigma_grad * self.loco_step_size

        weight = self.score_decay**ema
        weight = weight / weight.sum().clamp(min=1e-8)
        self._mean, self._sigma = weight @ target_mean, weight @ target_sigma


class ImplicitNaturalGradientSampler(BaseSampler):
    """
    Taken from https://github.com/optuna/optunahub-registry/blob/89da32cfc845c4275549000369282631c70bdaff/package/samplers/implicit_natural_gradient/sampler.py
    under the MIT License
    """

    def __init__(
        self,
        search_space: Dict[str, BaseDistribution],
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        lr: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ) -> None:
        self.search_space = search_space
        self._x0 = x0
        self._sigma0 = sigma0
        self._lr = lr
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._optimizer: Optional[FastINGO] = None
        self._seed = seed
        self._population_size = population_size

        self._param_queue: List[Dict[str, Any]] = []

    def _get_optimizer(self) -> FastINGO:
        assert self._optimizer is not None
        return self._optimizer

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._optimizer:
            self._optimizer._rng.seed()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self.search_space.items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def _check_trial_is_generation(self, trial: FrozenTrial) -> bool:
        current_gen = self._get_optimizer().generation
        trial_gen = trial.system_attrs.get("ingo", -1)
        return current_gen == trial_gen

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            self._warn_independent_sampling = False
            return {}

        trans = _SearchSpaceTransform(search_space)

        if self._optimizer is None:
            self._optimizer = self._init_optimizer(trans, population_size=self._population_size)

        if self._optimizer.dim != len(trans.bounds):
            self._warn_independent_sampling = False
            return {}

        solution_trials = [t for t in completed_trials if self._check_trial_is_generation(t)]
        for t in solution_trials:
            self._optimizer.tell(-t.value if study.direction == StudyDirection.MAXIMIZE else t.value)

        study._storage.set_trial_system_attr(trial._trial_id, "ingo", self._get_optimizer().generation)
        return trans.untransform(self._optimizer.ask())

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        population_size: Optional[int] = None,
    ) -> FastINGO:
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._x0 is None:
            mean = lower_bounds + (upper_bounds - lower_bounds) / 2
        else:
            mean = trans.transform(self._x0)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0
        inv_sigma = 1 / sigma0 * np.ones(n_dimension)

        return FastINGO(
            mean=mean,
            inv_sigma=inv_sigma,
            lower=lower_bounds,
            upper=upper_bounds,
            seed=self._seed,
            population_size=population_size,
            learning_rate=self._lr,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


class ThreadLocalSampler(threading.local):
    sampler: BaseSampler | None = None


def init_cmaes(study, seed, trials, search_space):
    trials.sort(key=lambda trial: trial.datetime_complete)
    return CmaEsSampler(seed=seed, source_trials=trials, lr_adapt=True)


def init_hebo(study, seed, trials, search_space):
    sampler = HEBOSampler(search_space=search_space, seed=seed)
    for trial in trials:
        sampler.after_trial(study, trial, TrialState.COMPLETE, trial.values)
    return sampler


def init_botorch(study, seed, trials, search_space):
    return BoTorchSampler(search_space=search_space, seed=seed, device="cuda")  # will automatically pull in latest data


def init_nsgaii(study, seed, trials, search_space):
    module = optunahub.load_module(
        "samplers/nsgaii_with_initial_trials",
    )
    return module.NSGAIIwITSampler(seed=seed)


def init_ingo(study, seed, trials, search_space):
    return ImplicitNaturalGradientSampler(search_space=search_space, seed=seed)


class AutoSampler(BaseSampler):
    def __init__(
        self,
        samplers: Iterable[Tuple[int, Callable]] | None = None,
        search_space: dict[str, BaseDistribution] = None,
        *,
        seed: int | None = None,
        constraints_func: None = None,
    ) -> None:
        assert constraints_func is None
        if samplers is None:
            samplers = ((0, init_halton), (100, init_nsgaii))
        self.sampler_indices = np.sort(np.array([x[0] for x in samplers], dtype=np.int32))
        self.samplers = [x[1] for x in sorted(samplers, key=lambda x: x[0])]
        self.search_space = search_space
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self._thread_local_sampler = ThreadLocalSampler()
        self._constraints_func = constraints_func
        self._completed_trials = 0
        self._current_index = -1

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_thread_local_sampler"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._thread_local_sampler = ThreadLocalSampler()

    @property
    def _sampler(self) -> BaseSampler:
        if self._thread_local_sampler.sampler is None:
            seed_for_random_sampler = self._rng.rng.randint(_MAXINT32)
            self._sampler = RandomSampler(seed=seed_for_random_sampler)

        return self._thread_local_sampler.sampler

    @_sampler.setter
    def _sampler(self, sampler: BaseSampler) -> None:
        self._thread_local_sampler.sampler = sampler

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._sampler.reseed_rng()

    def _update_sampler(self, study: Study):
        if len(study.directions) > 1:
            raise ValueError("Multi-objective optimization is not supported.")

        if isinstance(self._sampler, CmaEsSampler):
            return

        complete_trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        self._completed_trials = max(self._completed_trials, len(complete_trials))
        new_index = (self._completed_trials >= self.sampler_indices).sum() - 1
        if new_index == self._current_index:
            return
        self._current_index = new_index
        self._sampler = self.samplers[new_index](
            study, self._rng.rng.randint(_MAXINT32), complete_trials, self.search_space
        )

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler.sample_relative(study, trial, self.search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # NOTE(nabenabe): Sampler must be updated in this method. If, for example, it is updated in
        # infer_relative_search_space, the sampler for before_trial and that for sample_relative,
        # after_trial might be different, meaning that the sampling routine could be incompatible.
        if len(study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)) != 0:
            self._update_sampler(study)

        sampler_name = self._sampler.__class__.__name__
        study._storage.set_trial_system_attr(trial._trial_id, _SAMPLER_KEY, sampler_name)
        self._sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._sampler.after_trial(study, trial, state, values)


"""Hyperparameter sweeps with Halton sequences of quasi-random numbers.
Based off the algorithms described in https://arxiv.org/abs/1706.03200. Inspired
by the code in
https://github.com/google/uncertainty-baselines/blob/master/uncertainty_baselines/halton.py
written by the same authors.
"""
import collections
import functools
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from numpy import random
from optuna.samplers import BaseSampler
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.trial import FrozenTrial
from optuna.study import Study

_SweepSequence = List[Dict[str, Any]]
_GeneratorFn = Callable[[float], Tuple[str, float]]

def generate_primes(n: int) -> List[int]:
    """Generate primes less than `n` (except 2) using the Sieve of Sundaram."""
    half_m1 = int((n - 2) / 2)
    sieve = [0] * (half_m1 + 1)
    for outer in range(1, half_m1 + 1):
        inner = outer
        while outer + inner + 2 * outer * inner <= half_m1:
            sieve[outer + inner + (2 * outer * inner)] = 1
            inner += 1
    return [2 * i + 1 for i in range(1, half_m1 + 1) if sieve[i] == 0]

def _is_prime(n: int) -> bool:
    """Check if `n` is a prime number."""
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

def _generate_dim(num_samples: int,
                  base: int,
                  per_dim_shift: bool,
                  shuffled_seed_sequence: List[int]) -> List[float]:
    """Generate `num_samples` from a Van der Corput sequence with base `base`.
    Args:
        num_samples: int, the number of samples to generate.
        base: int, the base for the Van der Corput sequence. Must be prime.
        per_dim_shift: boolean, if true then each dim in the sequence is shifted by
            a random float (and then passed through fmod(n, 1.0) to keep in the range
            [0, 1)).
        shuffled_seed_sequence: An optional list of length `base`, used as the input
            sequence to generate samples. Useful for deterministic testing.
    Returns:
        A shuffled Van der Corput sequence of length `num_samples`, and optionally a
        shift added to each dimension.
    Raises:
        ValueError: if `base` is negative or not prime.
    """
    if base < 0 or not _is_prime(base):
        raise ValueError('Each Van der Corput sequence requires a prime `base`, '
                         f'received {base}.')
    rng = random.RandomState(base)
    if shuffled_seed_sequence is None:
        shuffled_seed_sequence = list(range(1, base))
        # np.random.RandomState uses MT19937 (see
        # https://numpy.org/devdocs/reference/random/legacy.html#numpy.random.RandomState).
        rng.shuffle(shuffled_seed_sequence)
        shuffled_seed_sequence = [0] + shuffled_seed_sequence
    
    # Optionally generate a random float in the range [0, 1) to shift this
    # dimension by.
    dim_shift = rng.random_sample() if per_dim_shift else None
    dim_sequence = []
    for i in range(1, num_samples + 1):
        num = 0.
        denominator = base
        temp_i = i
        while temp_i:
            num += shuffled_seed_sequence[temp_i % base] / denominator
            denominator *= base
            temp_i //= base
        if per_dim_shift:
            num = math.fmod(num + dim_shift, 1.0)
        dim_sequence.append(num)
    return dim_sequence

Matrix = List[List[int]]

def generate_sequence(num_samples: int,
                      num_dims: int,
                      skip: int = 100,
                      per_dim_shift: bool = True,
                      shuffle_sequence: bool = True,
                      primes: Sequence[int] = None,
                      shuffled_seed_sequence: Matrix = None) -> Matrix:
    """Generate `num_samples` from a Halton sequence of dimension `num_dims`.
    Each dimension is generated independently from a shuffled Van der Corput
    sequence with a different base prime, and an optional shift added. The
    generated points are, by default, shuffled before returning.
    Args:
        num_samples: int, the number of samples to generate.
        num_dims: int, the number of dimensions per generated sample.
        skip: non-negative int, if positive then a sequence is generated and the
            first `skip` samples are discarded in order to avoid unwanted
            correlations.
        per_dim_shift: boolean, if true then each dim in the sequence is shifted by
            a random float (and then passed through fmod(n, 1.0) to keep in the range
            [0, 1)).
        shuffle_sequence: boolean, if true then shuffle the sequence before
            returning.
        primes: An optional sequence (of length `num_dims`) of prime numbers to use
            as the base for the Van der Corput sequence for each dimension. Useful for
            deterministic testing.
        shuffled_seed_sequence: An optional list of length `num_dims`, with each
            element being a sequence of length `primes[d]`, used as the input sequence
            to the Van der Corput sequence for each dimension. Useful for
            deterministic testing.
    Returns:
        A shuffled Halton sequence of length `num_samples`, where each sample has
        `num_dims` dimensions, and optionally a shift added to each dimension.
    Raises:
        ValueError: if `skip` is negative.
        ValueError: if `primes` is provided and not of length `num_dims`.
        ValueError: if `shuffled_seed_sequence` is provided and not of length
            `num_dims`.
        ValueError: if `shuffled_seed_sequence[d]` is provided and not of length
            `primes[d]` for any d in range(num_dims).
    """
    if skip < 0:
        raise ValueError(f'Skip must be non-negative, received: {skip}.')
    if primes is not None and len(primes) != num_dims:
        raise ValueError(
            'If passing in a sequence of primes it must be the same length as '
            f'num_dims={num_dims}, received {primes} (len {len(primes)}).')
    if shuffled_seed_sequence is not None:
        if len(shuffled_seed_sequence) != num_dims:
            raise ValueError(
                'If passing in `shuffled_seed_sequence` it must be the same length '
                f'as num_dims={num_dims}, received {shuffled_seed_sequence} '
                f'(len {len(shuffled_seed_sequence)}).')
        for d in range(num_dims):
            if len(shuffled_seed_sequence[d]) != primes[d]:
                raise ValueError(
                    'If passing in `shuffled_seed_sequence` it must have element `{d}` '
                    'be a sequence of length `primes[{d}]`={expected}, received '
                    '{actual} (len {length})'.format(
                        d=d,
                        expected=primes[d],
                        actual=shuffled_seed_sequence[d],
                        length=len(shuffled_seed_sequence[d])))
    
    if primes is None:
        primes = []
        prime_attempts = 1
        while len(primes) < num_dims + 1:
            primes = generate_primes(1000 * prime_attempts)
            prime_attempts += 1
        primes = primes[-num_dims - 1:-1]
    
    # Skip the first `skip` points in the sequence because they can have unwanted
    # correlations.
    num_samples += skip
    halton_sequence = []
    for d in range(num_dims):
        if shuffled_seed_sequence is None:
            dim_shuffled_seed_sequence = None
        else:
            dim_shuffled_seed_sequence = shuffled_seed_sequence[d]
        dim_sequence = _generate_dim(
            num_samples=num_samples,
            base=primes[d],
            shuffled_seed_sequence=dim_shuffled_seed_sequence,
            per_dim_shift=per_dim_shift)
        dim_sequence = dim_sequence[skip:]
        halton_sequence.append(dim_sequence)
    
    # Transpose the 2-D list to be shape [num_samples, num_dims].
    halton_sequence = list(zip(*halton_sequence))
    
    # Shuffle the sequence.
    if shuffle_sequence:
        random.shuffle(halton_sequence)
    
    return halton_sequence

def _generate_double_point(name: str,
                           min_val: float,
                           max_val: float,
                           scaling: str,
                           halton_point: float) -> Tuple[str, float]:
    """Generate a float hyperparameter value from a Halton sequence point."""
    if scaling not in ['linear', 'log']:
        raise ValueError(
            'Only log or linear scaling is supported for floating point '
            f'parameters. Received {scaling}.')
    if scaling == 'log':
        # To transform from [0, 1] to [min_val, max_val] on a log scale we do:
        # min_val * exp(x * log(max_val / min_val)).
        rescaled_value = (
            min_val * math.exp(halton_point * math.log(max_val / min_val)))
    else:
        rescaled_value = halton_point * (max_val - min_val) + min_val
    return name, rescaled_value

def _generate_discrete_point(name: str,
                             feasible_points: Sequence[Any],
                             halton_point: float) -> Any:
    """Generate a discrete hyperparameter value from a Halton sequence point."""
    index = int(math.floor(halton_point * len(feasible_points)))
    # Ensure index is within bounds
    index = min(index, len(feasible_points) - 1)
    return name, feasible_points[index]

_DiscretePoints = collections.namedtuple('_DiscretePoints', 'feasible_points')

def discrete(feasible_points: Sequence[Any]) -> _DiscretePoints:
    return _DiscretePoints(feasible_points)

def interval(start: Union[int, float], end: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    return start, end

def loguniform(name: str, range_endpoints: Tuple[Union[int, float], Union[int, float]]) -> _GeneratorFn:
    min_val, max_val = range_endpoints
    return functools.partial(_generate_double_point,
                             name,
                             float(min_val),
                             float(max_val),
                             'log')

def uniform(name: str, search_points: Union[_DiscretePoints, Tuple[Union[int, float], Union[int, float]]]) -> _GeneratorFn:
    if isinstance(search_points, _DiscretePoints):
        return functools.partial(_generate_discrete_point,
                                 name,
                                 search_points.feasible_points)
    min_val, max_val = search_points
    return functools.partial(_generate_double_point,
                             name,
                             float(min_val),
                             float(max_val),
                             'linear')

def product(sweeps: Sequence[_SweepSequence]) -> _SweepSequence:
    """Cartesian product of a list of hyperparameter generators."""
    # A List[Dict] of hyperparameter names to sweep values.
    hyperparameter_sweep = []
    for hyperparameter_index in range(len(sweeps)):
        hyperparameter_sweep.append([])
        # Keep iterating until the iterator in sweep() ends.
        sweep_i = sweeps[hyperparameter_index]
        for point_index in range(len(sweep_i)):
            hyperparameter_name, value = list(sweep_i[point_index].items())[0]
            hyperparameter_sweep[-1].append((hyperparameter_name, value))
    return list(map(dict, itertools.product(*hyperparameter_sweep)))

def sweep(name, feasible_points: Sequence[Any]) -> _SweepSequence:
    return [{name: x} for x in feasible_points.feasible_points]

def zipit(generator_fns_or_sweeps: Sequence[Union[_GeneratorFn, _SweepSequence]],
          length: int) -> _SweepSequence:
    """Zip together a list of hyperparameter generators.
    Args:
        generator_fns_or_sweeps: A sequence of either:
            - Generator functions that accept a Halton sequence point and return a
            quasi-ranom sample, such as those returned by halton.uniform() or
            halton.loguniform()
            - Lists of dicts with one key/value such as those returned by
            halton.sweep()
            We need to support both of these (instead of having halton.sweep() return
            a list of generator functions) so that halton.sweep() can be used directly
            as a list.
        length: the number of hyperparameter points to generate. If any of the
            elements in generator_fns_or_sweeps are sweep lists, and their length is
            less than `length`, the sweep generation will be terminated and will be
            the same length as the shortest sweep sequence.
    Returns:
        A list of dictionaries, one for each trial, with a key for each unique
        hyperparameter name from generator_fns_or_sweeps.
    """
    halton_sequence = generate_sequence(
        num_samples=length, num_dims=len(generator_fns_or_sweeps))
    
    # A List[Dict] of hyperparameter names to sweep values.
    hyperparameter_sweep = []
    for trial_index in range(length):
        hyperparameter_sweep.append({})
        for hyperparameter_index in range(len(generator_fns_or_sweeps)):
            halton_point = halton_sequence[trial_index][hyperparameter_index]
            if callable(generator_fns_or_sweeps[hyperparameter_index]):
                generator_fn = generator_fns_or_sweeps[hyperparameter_index]
                hyperparameter_name, value = generator_fn(halton_point)
            else:
                sweep_list = generator_fns_or_sweeps[hyperparameter_index]
                if trial_index >= len(sweep_list):
                    break
                hyperparameter_point = sweep_list[trial_index]
                hyperparameter_name, value = list(hyperparameter_point.items())[0]
            hyperparameter_sweep[trial_index][hyperparameter_name] = value
    return hyperparameter_sweep

_DictSearchSpace = Dict[str, Dict[str, Union[str, float, Sequence]]]
_ListSearchSpace = List[Dict[str, Union[str, float, Sequence]]]


def generate_search(search_space: Union[_DictSearchSpace, _ListSearchSpace],
                    num_trials: int) -> List[collections.namedtuple]:
    """Generate a random search with the given bounds and scaling.
    Args:
        search_space: A dict where the keys are the hyperparameter names, and the
            values are a dict of:
                - {"min": x, "max": y, "scaling": z} where x and y are floats and z is
                one of "linear" or "log"
                - {"feasible_points": [...]} for discrete hyperparameters.
            Alternatively, it can be a list of dict where keys are the hyperparameter
            names, and the values are hyperparameters.
        num_trials: the number of hyperparameter points to generate.
    Returns:
        A list of length `num_trials` of namedtuples, each of which has attributes
        corresponding to the given hyperparameters, and values randomly sampled.
    """
    if isinstance(search_space, dict):
        all_hyperparameter_names = list(search_space.keys())
    elif isinstance(search_space, list):
        assert len(search_space) > 0
        all_hyperparameter_names = list(search_space[0].keys())
    else:
        raise AttributeError('tuning_search_space should either be a dict or list.')
    
    named_tuple_class = collections.namedtuple('Hyperparameters',
                                               all_hyperparameter_names)
    
    if isinstance(search_space, dict):
        hyperparameter_generators = []
        for name, space in search_space.items():
            if 'feasible_points' in space:  # Discrete search space.
                generator_fn = uniform(name, discrete(space['feasible_points']))
            else:  # Continuous space.
                if space['scaling'] == 'log':
                    generator_fn = loguniform(name, interval(space['min'], space['max']))
                else:
                    generator_fn = uniform(name, interval(space['min'], space['max']))
            hyperparameter_generators.append(generator_fn)
        return [
            named_tuple_class(**p)
            for p in zipit(hyperparameter_generators, num_trials)
        ]
    else:
        hyperparameters = []
        updated_num_trials = min(num_trials, len(search_space))
        if num_trials != len(search_space):
            print(f'num_trials was set to {num_trials}, but '
                  f'{len(search_space)} trial(s) found in the search space. '
                  f'Updating num_trials to {updated_num_trials}.')
        for trial in search_space:
            hyperparameters.append(named_tuple_class(**trial))
        return hyperparameters[:updated_num_trials]


class HaltonSampler(BaseSampler):
    """
    High-quality Halton quasi-random sampler using Google's implementation
    from "Critical Hyper-Parameters: No Random, No Cry"
    """
    
    def __init__(
        self,
        search_space: Dict[str, BaseDistribution],
        *,
        seed: Optional[int] = None,
        skip: int = 100,
        per_dim_shift: bool = True,
        shuffle_sequence: bool = True,
        batch_size: int = 1000,
    ) -> None:
        self.search_space = search_space
        self._seed = seed
        self._skip = skip
        self._per_dim_shift = per_dim_shift
        self._shuffle_sequence = shuffle_sequence
        self._batch_size = batch_size
        
        # Pre-generate parameter configurations
        self._param_configs = []
        self._current_batch = []
        self._batch_index = 0
        self._total_generated = 0
        self._initialize_generators()
    
    def _initialize_generators(self) -> None:
        """Convert Optuna distributions to Halton generators."""
        if not self.search_space:
            return
            
        # Filter valid parameters
        valid_params = {name: dist for name, dist in self.search_space.items() 
                       if not dist.single()}
        
        if not valid_params:
            return
            
        generators = []
        for name, dist in valid_params.items():
            if isinstance(dist, FloatDistribution):
                if dist.log:
                    gen = loguniform(name, interval(dist.low, dist.high))
                else:
                    gen = uniform(name, interval(dist.low, dist.high))
            elif isinstance(dist, IntDistribution):
                if dist.log:
                    gen = loguniform(name, interval(dist.low, dist.high))
                else:
                    # For integers, create discrete points
                    points = list(range(dist.low, dist.high + 1, dist.step or 1))
                    gen = uniform(name, discrete(points))
            elif isinstance(dist, CategoricalDistribution):
                gen = uniform(name, discrete(dist.choices))
            else:
                continue
                
            generators.append(gen)
        
        self._generators = generators
        self._generate_next_batch()
    
    def _generate_next_batch(self) -> None:
        """Generate next batch of samples."""
        if not self._generators:
            return
            
        # Set random seed for reproducibility
        if self._seed is not None:
            original_state = random.get_state()
            random.seed(self._seed + self._total_generated // self._batch_size)
        
        try:
            batch = zipit(self._generators, self._batch_size)
            self._current_batch = batch
            self._batch_index = 0
            self._total_generated += len(batch)
        finally:
            if self._seed is not None:
                random.set_state(original_state)
    
    def _get_next_sample(self) -> Dict[str, Any]:
        """Get next sample from current batch."""
        if self._batch_index >= len(self._current_batch):
            self._generate_next_batch()
        
        if not self._current_batch:
            return {}
            
        sample = self._current_batch[self._batch_index]
        self._batch_index += 1
        
        # Post-process for Optuna compatibility
        processed_sample = {}
        for name, value in sample.items():
            if name not in self.search_space:
                continue
                
            dist = self.search_space[name]
            
            if isinstance(dist, IntDistribution):
                if isinstance(value, float):
                    # Convert float to int and apply constraints
                    if dist.log:
                        value = int(round(value))
                    else:
                        value = int(round(value))
                    value = max(dist.low, min(dist.high, value))
                processed_sample[name] = int(value)
            elif isinstance(dist, FloatDistribution):
                if dist.step is not None:
                    # Apply step constraint
                    steps = round((value - dist.low) / dist.step)
                    value = dist.low + steps * dist.step
                value = max(dist.low, min(dist.high, value))
                processed_sample[name] = float(value)
            else:
                processed_sample[name] = value
                
        return processed_sample
    
    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self.search_space
    
    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return self._get_next_sample()
    
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Fallback to random sampling for independent parameters."""
        if self._seed is not None:
            rng = np.random.RandomState(self._seed + trial.number)
        else:
            rng = np.random.RandomState()
        
        if isinstance(param_distribution, FloatDistribution):
            if param_distribution.log:
                return float(np.exp(rng.uniform(
                    np.log(param_distribution.low), 
                    np.log(param_distribution.high)
                )))
            else:
                value = rng.uniform(param_distribution.low, param_distribution.high)
                if param_distribution.step is not None:
                    value = param_distribution.low + np.round(
                        (value - param_distribution.low) / param_distribution.step
                    ) * param_distribution.step
                return float(np.clip(value, param_distribution.low, param_distribution.high))
        
        elif isinstance(param_distribution, IntDistribution):
            if param_distribution.log:
                return int(np.exp(rng.uniform(
                    np.log(param_distribution.low), 
                    np.log(param_distribution.high + 1)
                )))
            else:
                return int(rng.randint(param_distribution.low, param_distribution.high + 1))
        
        elif isinstance(param_distribution, CategoricalDistribution):
            return rng.choice(param_distribution.choices)
        
        raise NotImplementedError(f"Unsupported distribution: {param_distribution}")

    def reseed_rng(self) -> None:
        """Reseed the random number generator."""
        if self._seed is not None:
            self._seed = np.random.randint(0, 2**31 - 1)
            self._initialize_generators()

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        """Called before each trial starts."""
        pass

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state,
        values: Optional[Sequence[float]],
    ) -> None:
        """Called after each trial completes."""
        pass


# Integration function for AutoSampler
def init_halton(study, seed, trials, search_space):
    """Initialize Halton sampler for use with AutoSampler."""
    return HaltonSampler(
        search_space=search_space,
        seed=seed,
        skip=100,  # Important for quality - skips first 100 correlated points
        per_dim_shift=True,  # Improves uniformity
        shuffle_sequence=True,  # Reduces correlations
        batch_size=1000  # Generate samples in batches for efficiency
    )