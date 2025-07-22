import functools
import math
from typing import Optional

import torch.optim

from . import chainable as C
from . import utils

class MARSWrapper(torch.optim.Optimizer):
    def __init__(self, params, wrapped_optimizer: utils.StatefulOptimizer, mars_gamma: float = 0.025):
        if not isinstance(wrapped_optimizer, utils.StatefulOptimizer):
            raise ValueError(f"{wrapped_optimizer.__class__.__name__} is not a StatefulOptimizer")
        super().__init__(params, {"mars_gamma": mars_gamma})
        self.wrapped_optimizer = wrapped_optimizer
        self.prev_params = None
        self.first_step = True
        
        # Add prev_grads and mars_gamma to wrapped optimizer's param groups
        for group in self.wrapped_optimizer.param_groups:
            group["prev_grads"] = []
            group["mars_gamma"] = mars_gamma

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("MARS requires closure")
        
        # First pass: current batch on current model
        with torch.enable_grad():
            loss = closure()
        
        # Second pass: current batch on previous model (if not first step)
        if not self.first_step:
            self._compute_and_store_prev_grads(closure)
        else:
            # First step: no previous model, so prev_grads remain empty
            self.first_step = False
        
        # Store current parameters for next iteration
        self._store_current_params()
        
        # Let wrapped optimizer do its update using current grads + stored prev_grads
        final_loss = self.wrapped_optimizer.step(closure)
            
        return final_loss

    def _compute_and_store_prev_grads(self, closure):
        """Compute gradients of current batch on previous model parameters"""
        if self.prev_params is None:
            return
        
        # Save current parameters and gradients
        current_params = []
        current_grads = []
        for group in self.param_groups:
            current_params.append([p.data.clone() for p in group["params"]])
            current_grads.append([p.grad.clone() if p.grad is not None else None 
                                for p in group["params"]])
        
        # Debug: Check parameter values before restoration
        param_sum_before = sum(torch.sum(p.data) for group in self.param_groups for p in group["params"])
        
        
        try:
            # Restore previous parameters
            param_idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if param_idx < len(self.prev_params):
                        p.data.copy_(self.prev_params[param_idx])
                    param_idx += 1
            
            # Debug: Check parameter values after restoration
            param_sum_after_restore = sum(torch.sum(p.data) for group in self.param_groups for p in group["params"])
            
            
            # Compute gradients on previous model with current batch
            self.wrapped_optimizer.zero_grad()
            with torch.enable_grad():
                prev_loss = closure()
                
            
            # Store prev_grads in wrapped optimizer's param groups
            for group, wrapped_group in zip(self.param_groups, self.wrapped_optimizer.param_groups):
                prev_grads = []
                for p in group["params"]:
                    if p.grad is not None:
                        prev_grads.append(p.grad.clone())
                wrapped_group["prev_grads"] = prev_grads
                
            
        finally:
            # Restore current parameters and gradients
            param_idx = 0
            for group_params, group_grads, group in zip(current_params, current_grads, self.param_groups):
                for p, current_param_data, current_grad in zip(group["params"], group_params, group_grads):
                    p.data.copy_(current_param_data)
                    p.grad = current_grad
                    param_idx += 1
            
            # Debug: Check parameter values after final restoration
            param_sum_final = sum(torch.sum(p.data) for group in self.param_groups for p in group["params"])
            

    def _store_current_params(self):
        """Store current parameters for next iteration"""
        self.prev_params = []
        for group in self.param_groups:
            for p in group["params"]:
                self.prev_params.append(p.data.clone().detach())

    def zero_grad(self, set_to_none: bool = True):
        self.wrapped_optimizer.zero_grad(set_to_none)


class SAMWrapper(torch.optim.Optimizer):
    def __init__(self, params, wrapped_optimizer: utils.StatefulOptimizer, ball: float = 0.1):
        if not isinstance(wrapped_optimizer, utils.StatefulOptimizer):
            raise ValueError(f"{wrapped_optimizer.__class__.__name__} is not a HeavyBall optimizer")
        super().__init__(params, {"ball": ball})
        self.wrapped_optimizer = wrapped_optimizer

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SAM requires closure")
        with torch.enable_grad():
            closure()
        old_params = [utils.sam_step(group["params"], group["ball"]) for group in self.param_groups]

        originaL_handle_closure = self.wrapped_optimizer._handle_closure

        def _handle_closure(closure):
            originaL_handle_closure(closure)
            for group, old in zip(self.param_groups, old_params):
                utils.copy_stochastic_list_(group["params"], old)

        try:
            self.wrapped_optimizer._handle_closure = _handle_closure
            loss = self.wrapped_optimizer.step(closure)
        finally:
            self.wrapped_optimizer._handle_closure = originaL_handle_closure
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.wrapped_optimizer.zero_grad()


class ForeachAdamWScheduled(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = True,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)


class ForeachSGD(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_SGD)


class ForeachAdamWEMA(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = True,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)


class ForeachAdamWEMAScheduled(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = True,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = True,
        use_ema: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)


class ForeachAdamW(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)


class ForeachAdamW_lr_schedule(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        lr_schedule=True,
        scheduler_type="onecycle",
        total_steps=100000,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adam)

class ForeachSTORM(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        requires_prev_minibatch = False,
        requires_prev_model = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_STORM, requires_prev_minibatch=requires_prev_minibatch, requires_prev_model=requires_prev_model)


class ForeachSTORM_plus(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        requires_prev_minibatch = False,
        requires_prev_model = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_STORM_plus, requires_prev_minibatch=requires_prev_minibatch, requires_prev_model=requires_prev_model)


class ForeachMARSAdamW(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        requires_prev_minibatch = False,
        requires_prev_model = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_MARSAdamW, requires_prev_minibatch=requires_prev_minibatch, requires_prev_model=requires_prev_model)


class ForeachMARSAdamWScheduled(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = True,
        use_ema: bool = False,
        requires_prev_minibatch = False,
        requires_prev_model = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_MARSAdamW, requires_prev_minibatch=requires_prev_minibatch, requires_prev_model=requires_prev_model)


class ForeachRMSprop(C.BaseOpt):
    """
    Debiased RMSprop (not torch.optim.RMSprop)
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
        )


class ForeachSFAdamWEMA(C.ScheduleFree):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = True,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
            C.update_by_schedule_free,
        )


class ForeachSFAdamW(C.ScheduleFree):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        mars_schedule: bool = False,
        use_ema: bool = False,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
            C.update_by_schedule_free,
        )


class MSAMLaProp(C.MSAM):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        sam_step_size: float = 0.1,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_exp_avg_sq,
            C.update_by_msam,
        )


class PaLMForeachSFAdamW(ForeachSFAdamW):
    palm: bool = True


class ForeachADOPT(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_adopt)


class ForeachMuon(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        nesterov: bool = True,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.nesterov_ema if nesterov else C.exp_avg,
            C.orthogonalize_update,
        )


class ForeachLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(params, defaults, foreach, gradient_clipping, update_clipping, palm, C.update_by_laprop)


class MuonLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.orthogonalize_update,
        )


class ForeachSOAP(C.BaseOpt):
    """
    ForeachSOAP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        stochastic_schedule: bool = False,
        precond_grad_accum: bool = False,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  #
            C.scale_by_soap,
        )


class ForeachSignLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.sign,
        )


class ForeachCOSMOS(C.BaseOpt):
    """
    ForeachCOSMOS (Combination of SOAP and Muon)

    Sources:
        https://github.com/lliu606/COSMOS/blob/main/COSMOS.py
        https://arxiv.org/html/2502.17410v1
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.1,  # use shampoo_beta as gamma in cosmos
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        stochastic_schedule: bool = False,
        precond_grad_accum: bool = False,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  #
            C.scale_by_cosmos,
        )


class ForeachSignLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.sign,
        )


class ForeachSOLP(C.BaseOpt):
    """
    ForeachSOLP

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        stochastic_schedule: bool = False,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  #
            functools.partial(C.scale_by_soap, inner="laprop"),
        )



class ForeachSPlus(C.BaseOpt):
    """
    ForeachSPlus

    Sources:
        Baseline SOAP:
            SOAP: Improving and Stabilizing Shampoo using Adam
            Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
            https://arxiv.org/abs/2409.11321
            https://github.com/nikhilvyas/SOAP

        SPlus:
            A Stable Whitening Optimizer for Efficient Neural Network Training
            Kevin Frans, Sergey Levine, Pieter Abbeel
            https://arxiv.org/pdf/2506.07254
    """

    use_precond_schedule: bool = False

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 2,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 0,
        split: bool = False,
        foreach: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        palm: bool = C.use_default,
        precond_scheduler=(1 / 3, 9),
        beta2_scale: float = 0.8,
        use_precond_schedule: bool = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        storage_dtype: str = "float32",
        stochastic_schedule: bool = False,
        **kwargs,
    ):
        use_precond_schedule = C.default(use_precond_schedule, self.use_precond_schedule)

        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        if use_precond_schedule:
            del defaults["precondition_frequency"]
            self.precond_schedule = utils.get_soap_precond_schedule(defaults.pop("precond_scheduler"))
        else:
            del defaults["precond_scheduler"]
            self.precond_schedule = 1 / defaults.pop("precondition_frequency")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,  
            functools.partial(C.scale_by_soap, inner="lion"),
        )


class PaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = False
    palm: bool = True


class PrecondScheduleForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True


class PrecondSchedulePaLMForeachSOAP(ForeachSOAP):
    use_precond_schedule: bool = True
    palm: bool = True


class OrthoLaProp(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.orthogonalize_grad_to_param,
            C.scale_by_laprop,
        )


class LaPropOrtho(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        foreach: bool = True,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        beta2_scale: float = 0.8,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")
        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            palm,
            C.scale_by_laprop,
            C.orthogonalize_grad_to_param,
        )


class ForeachPSGDKron(C.BaseOpt):
    """
    Originally from Evan Walters and Omead Pooladzandi, 2024
    Modified under Creative Commons Attribution 4.0 International
    Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    delayed: bool = False
    cached: bool = False
    exp_avg_input: bool = True
    quad: bool = False

    def __init__(
        self,
        params,
        lr=0.001,
        beta=None,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=2048,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        warmup_steps: int = 0,
        merge_dims: bool = False,
        split: bool = False,
        store_triu_as_line: bool = True,
        foreach: bool = True,
        q_dtype="float32",
        stochastic_schedule: bool = False,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: Optional[bool] = C.use_default,
        cached: Optional[bool] = C.use_default,
        exp_avg_input: Optional[bool] = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,  #
        adaptive: bool = False,
        ortho_method: Optional[str] = None,  # If None, no orthogonalization
        precond_grad_accum: bool = False,
        lower_bound_beta: float = 0.9,  # 0.0 recovers pre-2.0.0 PSGD
        inverse_free: bool = C.use_default,
        dampening: float = 2**-13,
        precond_update_power_iterations: int = 2,
        # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: Optional[float] = None,
        precond_lr: float = 0.1,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        cached = C.default(cached, self.cached)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)
        inverse_free = C.default(inverse_free, self.quad)

        defaults = locals()
        defaults.pop("self")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        self.precond_schedule = (
            defaults.pop("preconditioner_update_probability") or utils.precond_update_prob_schedule()
        )
        params = defaults.pop("params")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            *(C.exp_avg,) * exp_avg_input,  #
            functools.partial(C.scale_by_delayed_psgd if delayed else C.scale_by_psgd, cached=cached),
        )


class ForeachPurePSGD(ForeachPSGDKron):
    exp_avg_input: bool = False


class ForeachCachedDelayedPSGDKron(ForeachPSGDKron):
    delayed: bool = True
    cached: bool = True


class ForeachCachedPSGDKron(ForeachPSGDKron):
    cached: bool = True


class ForeachDelayedPSGD(ForeachPSGDKron):
    delayed: bool = True


class ForeachCachedNewtonPSGD(ForeachCachedPSGDKron):
    hessian_approx = True


class NewtonHybrid2PSGDKron(ForeachCachedNewtonPSGD):
    hvp_interval = 2


class QUAD(ForeachPSGDKron):
    quad = True
    cached = True


class ForeachPSGDLRA(C.BaseOpt):
    """
    Originally from Evan Walters and Omead Pooladzandi, 2024
    Modified under Creative Commons Attribution 4.0 International
    Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
    """

    delayed: bool = False
    exp_avg_input: bool = True

    def __init__(
        self,
        params,
        lr=0.001,
        beta=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        momentum_into_precond_update=True,
        rank: Optional[int] = None,
        warmup_steps: int = 0,
        foreach: bool = True,
        q_dtype="float32",
        stochastic_schedule: bool = False,
        storage_dtype: str = "float32",
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        delayed: Optional[bool] = C.use_default,
        exp_avg_input: Optional[bool] = C.use_default,
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        eps: float = 1e-8,  #
        precond_grad_accum: bool = False,  # expert parameters
        precond_init_scale=None,
        precond_init_scale_scale: float = 1,
        precond_init_scale_power: Optional[float] = None,
        precond_lr: float = 0.1,
        **kwargs,
    ):
        delayed = C.default(delayed, self.delayed)
        exp_avg_input = C.default(exp_avg_input, self.exp_avg_input)
        update_clipping = C.default(update_clipping, utils.trust_region_clip_)

        defaults = locals()
        defaults.pop("self")
        defaults.update(defaults.pop("kwargs"))

        if kwargs:
            utils.warn_once(f"Working with uncaptured keyword arguments: {kwargs}")

        self.precond_schedule = (
            defaults.pop("preconditioner_update_probability") or utils.precond_update_prob_schedule()
        )
        params = defaults.pop("params")

        if rank is None:
            utils.warn_once(
                f"{rank=}. It will be set to log2(param_count). This requires `params` to be of type list. Currently, {type(params)=}"
            )
            params = list(params)
            defaults["rank"] = round(math.log2(sum(p.numel() for p in params)))
            utils.warn_once(f"rank was set to {defaults['rank']}")

        super().__init__(
            params,
            defaults,
            foreach,
            gradient_clipping,
            update_clipping,
            False,  #
            *(C.exp_avg,) * exp_avg_input,  #
            C.scale_by_delayed_psgd_lra if delayed else C.scale_by_psgd_lra,
        )


class ForeachDelayedPSGDLRA(ForeachPSGDLRA):
    delayed: bool = True


class ForeachNewtonPSGDLRA(ForeachPSGDLRA):
    hessian_approx = True


class NewtonHybrid2PSGDLRA(ForeachNewtonPSGDLRA):
    hvp_interval = 2
    
def create_mars_wrapped_optimizer(base_optimizer_class, mars_gamma=0.025):
    """Factory function to create MARS-wrapped versions of optimizers"""
    
    class MARSWrappedOptimizer:
        def __init__(self, params, mars_gamma=mars_gamma, **kwargs):
            params = list(params)
            base_optimizer = base_optimizer_class(params, **kwargs)
            self.optimizer = MARSWrapper(params, base_optimizer, mars_gamma=mars_gamma)
            
            # Store initial parameter checksums
            self.param_checksums = []
            for p in params:
                self.param_checksums.append(torch.sum(p).item())
            
        def step(self, closure=None):
            # Check if parameters were reset since last step
            current_checksums = []
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    current_checksums.append(torch.sum(p).item())
            
            if hasattr(self, 'last_checksums'):
                if current_checksums == self.param_checksums:
                    print("WARNING: Parameters were reset to initial values!")
            
            result = self.optimizer.step(closure)
            
            # Store checksums after update
            self.last_checksums = []
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    self.last_checksums.append(torch.sum(p).item())
            
            return result
            
        def zero_grad(self, set_to_none: bool = True):
            return self.optimizer.zero_grad(set_to_none)
        
        def state_dict(self):
            return self.optimizer.state_dict()
        
        def load_state_dict(self, state_dict):
            return self.optimizer.load_state_dict(state_dict)
        
        @property
        def param_groups(self):
            return self.optimizer.param_groups
        
        @param_groups.setter
        def param_groups(self, value):
            self.optimizer.param_groups = value
    
    return MARSWrappedOptimizer

class ForeachDistributedShampoo(C.BaseOpt):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-12,
        weight_decay=0,
        max_preconditioner_dim=1024,
        precondition_frequency=20,
        start_preconditioning_step=101,
        use_merge_dims=True,
        merge_small_dims_threshold=4096,
        preconditioner_type="all",  # "all", "input", "output" 
        skip_preconditioning_rank_lt=1,
        skip_preconditioning_dim_size_gt=0,  # 0 = no limit
        grafting_type="adam",  # "adam", "rmsprop", "sgd", "none"
        foreach: bool = True,
        storage_dtype: str = "float32",
        gradient_clipping: C.str_or_fn = C.use_default,
        update_clipping: C.str_or_fn = C.use_default,
        palm: bool = C.use_default,
        mars: bool = False,
        caution: bool = False,
        **kwargs,
    ):
        defaults = locals()
        defaults.pop("self")
        params = defaults.pop("params")
        defaults.update(defaults.pop("kwargs"))
        
        super().__init__(
            params, 
            defaults, 
            foreach, 
            gradient_clipping,
            update_clipping,
            palm,
            C.update_by_tensor_shampoo
        )
        

# Then create the specific optimizer
MARSWrappedAdamW = create_mars_wrapped_optimizer(ForeachAdamW)
PalmForEachSoap = PaLMForeachSOAP
PaLMSOAP = PaLMForeachSOAP
PaLMSFAdamW = PaLMForeachSFAdamW
SOAP = ForeachSOAP
SFAdamW = ForeachSFAdamW
LaProp = ForeachLaProp
ADOPT = ForeachADOPT
RMSprop = ForeachRMSprop
PrecondScheduleSOAP = PrecondScheduleForeachSOAP
PrecondSchedulePaLMSOAP = PrecondSchedulePaLMForeachSOAP
PSGDKron = ForeachPSGDKron
AdamW = ForeachAdamW
PurePSGD = ForeachPurePSGD
DelayedPSGD = ForeachDelayedPSGD
CachedPSGDKron = ForeachCachedPSGDKron
CachedDelayedPSGDKron = ForeachCachedDelayedPSGDKron
Muon = ForeachMuon
SignLaProp = ForeachSignLaProp
DelayedPSGDLRA = ForeachDelayedPSGDLRA
PSGDLRA = ForeachPSGDLRA
NewtonPSGDLRA = ForeachNewtonPSGDLRA
AdamWScheduled = ForeachAdamWScheduled
AdamWEMA = ForeachAdamWEMA
AdamWEMAScheduled = ForeachAdamWEMAScheduled
SFAdamWEMA = ForeachSFAdamWEMA
STORM = ForeachSTORM
STORM_plus = ForeachSTORM_plus
MARSAdamW = ForeachMARSAdamW
MARSAdamWScheduled = ForeachMARSAdamWScheduled
SGD = ForeachSGD
SPlus = ForeachSPlus
AdamW_lr_schedule = ForeachAdamW_lr_schedule
COSMOS = ForeachCOSMOS
DistributedShampoo = ForeachDistributedShampoo
__all__ = [
    "Muon",
    "RMSprop",
    "PrecondSchedulePaLMSOAP",
    "PSGDKron",
    "PurePSGD",
    "DelayedPSGD",
    "CachedPSGDKron",
    "CachedDelayedPSGDKron",
    "PalmForEachSoap",
    "PaLMSOAP",
    "PaLMSFAdamW",
    "LaProp",
    "ADOPT",
    "PrecondScheduleSOAP",
    "PrecondSchedulePaLMSOAP",
    "RMSprop",
    "MuonLaProp",
    "ForeachSignLaProp",
    "ForeachDelayedPSGDLRA",
    "ForeachPSGDLRA",
    "ForeachPSGDLRA",
    "ForeachNewtonPSGDLRA",  
    "ForeachAdamW",
    "ForeachSFAdamW",
    "ForeachLaProp",
    "ForeachADOPT",
    "ForeachSOAP",
    "ForeachPSGDKron",
    "ForeachPurePSGD",
    "ForeachDelayedPSGD",
    "ForeachCachedPSGDKron",
    "ForeachCachedDelayedPSGDKron",
    "ForeachRMSprop",
    "ForeachMuon",
    "ForeachCachedNewtonPSGD",
    "OrthoLaProp",
    "LaPropOrtho",
    "SignLaProp",
    "DelayedPSGD",
    "PSGDLRA",
    "NewtonPSGDLRA",
    "NewtonHybrid2PSGDLRA",
    "NewtonHybrid2PSGDKron",
    "MSAMLaProp",
    "ForeachAdamWScheduled",
    "ForeachAdamWEMA",
    "ForeachAdamWEMAScheduled",
    "ForeachSFAdamWEMA",
    "ForeachSTORM",
    "ForeachMARSAdamW",
    "ForeachMARSAdamWScheduled",
    "ForeachSTORM_plus",
    "ForeachSGD",
    "ForeachSPlus",
    "ForeachMARSWrappedAdamW",
    "ForeachAdamW_lr_schedule",
    "ForeachCOSMOS"
    "ForeachDistributedShampoo"
]
