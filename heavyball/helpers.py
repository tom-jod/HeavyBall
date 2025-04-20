import functools
import math
from typing import Any, Callable, Sequence, Tuple, Union

import numpy
import numpy as np
import torch
from botorch.utils.sampling import manual_seed
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study, StudyDirection
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


class BoTorchSampler(BaseSampler):
    """
    A significantly more efficient implementation of `BoTorchSampler` from Optuna - keeps more on the GPU / in torch
    The original is available at https://github.com/optuna/optuna-integration/blob/156a8bc081322791015d2beefff9373ed7b24047/optuna_integration/botorch/botorch.py under the MIT License
    The original API is kept for backward compatibility, but many arguments are ignored to improve maintainability.
    """

    def __init__(
        self,
        *,
        candidates_func: None = None,
        constraints_func: None = None,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
        independent_sampler: None = None,
        seed: int | None = None,
        device: torch.device | None = None,
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
        self._search_space = IntersectionSearchSpace()
        self._device = device or torch.device("cpu")
        self.seen_trials = set()
        self._values = None
        self._params = None
        self._index = 0

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

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
