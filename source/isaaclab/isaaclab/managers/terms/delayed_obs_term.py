# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils.buffers import DelayBuffer


class DelayedObsTerm(ManagerTermBase):
    """Return a stochastically delayed (stale) version of another observation term.
    This can also be used to model multi-rate observations for non-sensor terms,
    e.g., pure MDP terms or proprioceptive terms.

    This wraps an existing observation term/function, pushes each new batched
    observation into a DelayBuffer, and returns an older sample according to a
    per-environment integer time-lag. Lags are drawn in [min_lag, max_lag].
    with an optional probability to *hold* the previous lag (to mimic repeated 
    frames). With 'update_period>0' (multi-rate), new lags are applied only
    on refresh ticks, which occur every update_period. Between refreshes the
    realised lag can increase at most by +1 (frame hold). This process is
    causal: the lag for each environment can only increase by 1 each step,
    ensuring that the returned observation is never older than the previous
    step's lagged observation.

    Shapes are preserved: the returned tensor has the exact shape of the wrapped
    term (``[num_envs, *obs_shape]``).

    Configuration (required nesting)
    --------------------------------
    Isaac Lab's manager **requires** class-based term params to be nested under
    the "_" key.
    
    Param keys:
        func (callable): The observation function to wrap. Must be callable
            with signature ``func(env, **func_params) -> torch.Tensor`` returning
            a batched tensor of shape ``[num_envs, ...]``.
        func_params (dict): Optional dict of keyword args to pass to `func`.
        min_lag (int): Minimum time-lag (in steps) to sample. Default 0.
        max_lag (int): Maximum time-lag (in steps) to sample. Default 3.
        per_env (bool): If True, sample a different lag for each environment.
            If False, use the same lag for all envs. Default True.
        hold_prob (float): Probability in [0, 1] of holding the previous lag
            instead of sampling a new one. Default 0.0 (always sample new).
        update_period (int): If > 0, apply new lags every `update_period`
            policy steps (models a lower sensor cadence). Between updates, the
            lag can increase by at most +1 each step (frame hold). If 0 (default),
            update every step.
        per_env_phase (bool): Only relevant if `update_period > 0`. If True,
            each environment has a different random phase offset for lag updates.
            If False, all envs update their lag simultaneously. Default True.

    Minimal example (drop in replacement for locomotion velocity tasks in velocity_env_cfg.py):
    Delay 1-6 steps, per-env, with 66% hold probability (no multi-rate):
        height_scan = ObsTerm(
            func=DelayedObsTerm,
            params={
                "_": {
                    "func": mdp.height_scan,
                    "func_params": {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    "min_lag": 1, "max_lag": 6,
                    "per_env": True, "hold_prob": 0.66,
                    "update_period": 0, "per_env_phase": False,
                },
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        
    No delay, multi-rate example (3-step cadence):
    height_scan = ObsTerm(
            func=DelayedObsTerm,
            params={
                "_": {
                    "func": mdp.height_scan,
                    "func_params": {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    "min_lag": 3, "max_lag": 3,
                    "per_env": True, "hold_prob": 0.0,
                    "update_period": 3, "per_env_phase": True,
                },
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        
    """

    def __init__(self, env, cfg: ObservationTermCfg):
        super().__init__(cfg, env)
        p_all = cfg.params or {}
        p = p_all.get("_", p_all)
        
        # The observation function we wrap
        self._func = p.get("func", None)
        if self._func is None or not callable(self._func):
            raise ValueError("DelayedObsTerm: `params.func` must be a callable returning a tensor [N, ...].")
        self._func_params = p.get("func_params", {}) or {}

        # Delay parameters
        self._min_lag = int(p.get("min_lag", 0))
        self._max_lag = int(p.get("max_lag", 3))
        self._per_env  = bool(p.get("per_env", True))
        self._hold_prob = float(p.get("hold_prob", 0.0))
        if self._min_lag < 0 or self._max_lag < self._min_lag:
            raise ValueError("DelayedObsTerm: require 0 <= min_lag <= max_lag.")
        if not (0.0 <= self._hold_prob <= 1.0):
            raise ValueError("DelayedObsTerm: `hold_prob` must be in [0, 1].")
        
        # multi-rate parameters
        self._update_period = int(p.get("update_period", 0))  # 0 = disabled
        if self._update_period < 0:
            raise ValueError("DelayedObsTerm: `update_period` must be non-negative.")
        elif self._update_period > 0 and self._update_period > self._max_lag:
            raise ValueError("DelayedObsTerm: `update_period` must be less than or equal to `max_lag`.")
        self._per_env_phase = bool(p.get("per_env_phase", True))
        self._step = 0
        self._phases = None  # [N], set on first call

        # State
        self._buf: Optional[DelayBuffer] = None
        self._last_lags: Optional[torch.Tensor] = None  # [N] 
        self._prev_realised_lags: Optional[torch.Tensor] = None

        # Do first call and prefill buffer
        with torch.no_grad():
            x0 = self._call_underlying()
        if x0.dim() < 1:
            raise RuntimeError("DelayedObsTerm: underlying func must return a tensor [N, ...].")

        # history_length = max_lag + 1 so lags in [0, max_lag] are addressable
        self._buf = DelayBuffer(
            history_length=self._max_lag + 1, batch_size=env.num_envs, device=str(x0.device)
        )
        # Prefill so early delays return valid data.
        for _ in range(self._max_lag + 1):
            self._buf.compute(x0)

    def reset(self, env_ids: Optional[Union[torch.Tensor, list]] = None):
        if self._buf is None:
            return
        # Full reset
        if env_ids is None:
            self._buf.reset()
            self._last_lags = None
            self._prev_realised_lags = None
            self._phases = None
            with torch.no_grad():
                x = self._call_underlying()
                for _ in range(self._max_lag + 1):
                    self._buf.compute(x)
        # Partial reset
        else:
            ids = torch.as_tensor(env_ids, device=self._prev_realised_lags.device if self._prev_realised_lags is not None else 'cpu', dtype=torch.long)
            # DelayBuffer does not support partial per env reset, instead
            # we force the lag to zero for these envs (no delay) so they immediately
            # read the latest observation. Buffer history is preserved.
            if self._prev_realised_lags is not None:
                self._prev_realised_lags[ids] = 0
            if self._last_lags is not None:
                self._last_lags[ids] = 0

    def __call__(self, env, **_) -> torch.Tensor:
        with torch.no_grad():
            self._step += 1
            x = self._call_underlying()
            # Sample lags (uniform in [min, max]; optional hold with probability hold_prob)
            lags = self._sample_uniform_lags_causal(batch_size=x.shape[0], device=x.device)
            # Set per-env (or shared) lag and get stale data
            self._buf.set_time_lag(lags)
            out = self._buf.compute(x)
            return out


    def _call_underlying(self) -> torch.Tensor:
        x = self._func(self._env, **self._func_params)
        if not isinstance(x, torch.Tensor):
            raise TypeError("DelayedObsTerm: underlying func must return a tensor [N, ...].")
        return x

    def _sample_uniform_lags_causal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # init phases on first use
        if self._update_period > 0 and self._phases is None:
            if self._per_env_phase:
                self._phases = torch.randint(0, self._update_period, (batch_size,), device=device)
            else:
                self._phases = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # draw desired lag in [min_lag, max_lag]
        if self._min_lag == self._max_lag:
            desired_lags = torch.full((batch_size,), self._max_lag, dtype=torch.long, device=device)
        else:
            desired_lags = torch.randint(self._min_lag, self._max_lag + 1, (batch_size,), device=device)

        # if not per-env, make all the lags the same
        if not self._per_env:
            desired_lags = torch.full_like(desired_lags, desired_lags[0])

        # optional “hold” (reuse previous realised lag) to mimic frame repeats
        if self._hold_prob > 0.0 and self._prev_realised_lags is not None:
            hold_mask = torch.rand((batch_size,), device=device) < self._hold_prob
            desired_lags = torch.where(hold_mask, self._prev_realised_lags, desired_lags)

        # If multi-rate, only update lag on certain steps
        if self._update_period > 0:
            # refresh exactly when (step - phase) % period == 0
            refresh_mask = ((self._step - self._phases) % self._update_period) == 0
            if self._prev_realised_lags is None:
                realised_lags = desired_lags  # first call
            else:
                # hold between refreshes: lag can increase by at most +1
                hold_realised_lags = (self._prev_realised_lags + 1).clamp(max=self._max_lag)
                # On refresh ticks: jump to desired (new frame arrived)
                realised_lags = torch.where(refresh_mask, desired_lags, hold_realised_lags)
        # not multi-rate: update every step
        else:
            # CAUSAL CLAMP: cannot jump back to frames older than last used
            if self._prev_realised_lags is not None:
                realised_lags = torch.minimum(desired_lags, self._prev_realised_lags + 1)
            else:
                realised_lags = desired_lags

        self._prev_realised_lags = realised_lags
        return realised_lags