"""Opt-in noise, history and sampler-sharing experiments. Ordinary math stays upstream."""

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator


class CircuitBendOptions(BaseModel):
    enabled: bool = False
    mode: Literal["noise", "plms_history", "ddim", "sampler_mix"] = "noise"
    mix_method: Literal["alternate", "coupled"] = "alternate"
    mix_first: Literal["plms", "ddim"] = "plms"
    mix_block: int = Field(1, ge=1, le=1000)
    mix_history: Literal["shared", "stale", "reset"] = "shared"
    mix_exchange: float = Field(0.5, ge=0, le=1)
    mix_output: Literal["plms", "ddim", "blend"] = "blend"
    mix_curve: Literal["constant", "progressive", "oscillate"] = "constant"
    mix_period: int = Field(8, ge=1, le=1000)
    schedule: Literal["native", "exact"] = "native"
    eta: float = Field(0.5, ge=0, le=1)
    history_source: Literal["freeze", "seed"] = "freeze"
    blend_target: Literal["history", "trajectory", "independent"] = "history"
    donor_seed: int = Field(3, ge=0, le=4294967295)
    blend_curve: Literal["constant", "progressive", "oscillate", "interlace"] = "constant"
    blend_start: float = Field(0, ge=0, le=1)
    blend_end: float = Field(1, ge=0, le=1)
    blend_period: int = Field(4, ge=1, le=1000)
    churn: float = Field(10.0, ge=0, le=100)
    noise: float = Field(1.0, ge=0, le=4)
    start: float = Field(1, ge=0)
    end: Optional[float] = Field(None, ge=0)
    window_unit: Literal["iterations", "percent"] = "iterations"
    every: int = Field(1, ge=1, le=1000)
    seed: int = Field(0, ge=0, le=4294967295)
    pattern: Literal["fresh", "held"] = "fresh"
    # None preserves recipes saved before the continuous memory control existed.
    memory: Optional[float] = Field(None, ge=0, le=1)
    fade_in: float = Field(0, ge=0, le=100)
    fade_out: float = Field(0, ge=0, le=100)
    history_strength: float = Field(1, ge=0, le=1)

    class Config:
        extra = "forbid"
        allow_inf_nan = False

    @validator("every", "seed", "donor_seed", "blend_period", "mix_block", "mix_period", pre=True)
    def integer_controls(cls, value):
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError("Circuit bend pulse interval and seed must be integers")
        return value

    @root_validator(skip_on_failure=True)
    def ordered_window(cls, values):
        if values.get("end") is not None and values["end"] < values["start"]:
            raise ValueError("Circuit bend end must be at or after start")
        for key in ("start", "end"):
            value = values.get(key)
            if value is None:
                continue
            if values["window_unit"] == "iterations" and (value < 1 or value != int(value)):
                raise ValueError("Circuit bend iteration boundaries must be positive integers")
            if values["window_unit"] == "percent" and value > 100:
                raise ValueError("Circuit bend percentage boundaries must be within 0..100")
        return values

    @property
    def active(self):
        if self.mode == "sampler_mix":
            return self.enabled
        effect = (self.history_strength > 0 if self.mode == "plms_history" else
                  self.eta > 0 if self.mode == "ddim" else self.churn > 0)
        return self.enabled and (effect or (self.mode != "noise" and self.schedule == "exact"))

    def blend_weight(self, iteration, steps):
        first, last = self.resolve_window(steps)
        offset = iteration - first
        if self.blend_curve == "constant":
            mix = self.blend_end
        else:
            if self.blend_curve == "progressive":
                phase = min(1, max(0, offset / max(1, last - first)))
            elif self.blend_curve == "oscillate":
                phase = (1 - math.cos(2 * math.pi * offset / self.blend_period)) / 2
            else:
                # Period is the number of iterations in each A/B block.
                phase = (offset // self.blend_period) % 2
            mix = self.blend_start + phase * (self.blend_end - self.blend_start)
        return self.history_strength * mix * self.weight(iteration, steps)

    def exchange_weight(self, iteration, steps):
        first, last = self.resolve_window(steps)
        phase = (min(1, max(0, (iteration - first) / max(1, last - first))) if self.mix_curve == "progressive"
                 else (1 - math.cos(2 * math.pi * (iteration - first) / self.mix_period)) / 2
                 if self.mix_curve == "oscillate" else 1)
        return self.mix_exchange * phase * self.weight(iteration, steps)

    @property
    def noise_memory(self):
        return self.memory if self.memory is not None else (1.0 if self.pattern == "held" else 0.0)

    def resolve_window(self, steps):
        if self.window_unit == "percent":
            first = max(1, math.ceil(steps * self.start / 100))
            last = max(1, math.ceil(steps * (100 if self.end is None else self.end) / 100))
            return first, last
        return int(self.start), steps if self.end is None else int(self.end)

    def weight(self, iteration, steps):
        """Inclusive 1-based window; linear fades as percentages of its span."""
        first, last = self.resolve_window(steps)
        if not first <= iteration <= last or (iteration - first) % self.every:
            return 0.0
        if first == last:
            return 1.0
        position = (iteration - first) / (last - first)
        incoming = min(1.0, position * 100 / self.fade_in) if self.fade_in else 1.0
        outgoing = min(1.0, (1 - position) * 100 / self.fade_out) if self.fade_out else 1.0
        return max(0.0, min(incoming, outgoing))

    def validate_render(self, sampler, steps, classic=True, txt2img=True):
        if not self.enabled:
            return
        supported = {"plms", "ddim"} if self.mode == "sampler_mix" else {"plms"} if self.mode == "plms_history" else {"ddim"} if self.mode == "ddim" else {"euler", "heun"}
        if not classic or not txt2img or sampler not in supported:
            raise ValueError("Circuit bend noise supports classic txt2img Euler and Heun; history bending supports PLMS; eta supports DDIM; sampler mixing supports PLMS/DDIM")
        first, last = self.resolve_window(steps)
        if first > steps or last > steps:
            raise ValueError("Circuit bend iteration window exceeds Inference Steps")


class ExperimentalSamplingStopped(RuntimeError):
    def __init__(self, latent, completed_steps):
        super().__init__("Circuit bend produced non-finite values; retained the last finite solver state")
        self.latent = latent
        self.completed_steps = completed_steps


class NoiseMemory:
    """Stationary Gaussian AR(1) field, with exact fresh and held endpoints."""

    def __init__(self, reference, seed, memory):
        import torch
        self.reference = reference
        self.generator = torch.Generator(device=reference.device).manual_seed(seed)
        self.memory = memory
        self.previous = None

    def next(self):
        import torch
        if self.memory == 1 and self.previous is not None:
            return self.previous
        fresh = torch.randn(self.reference.shape, dtype=self.reference.dtype,
                            device=self.reference.device, generator=self.generator)
        if self.previous is not None and self.memory > 0:
            fresh = self.memory * self.previous + math.sqrt(1 - self.memory ** 2) * fresh
        self.previous = fresh
        return fresh


class PLMSHistoryFreeze:
    """Blend an owned frozen history into updates without changing live history."""

    def __init__(self, options, steps):
        self.options = options
        self.steps = steps
        self.frozen = None
        self.frozen_at = None

    def apply(self, history, iteration):
        first, last = self.options.resolve_window(self.steps)
        selected = first <= iteration <= last and (iteration - first) % self.options.every == 0
        # Wait for native three-entry history, so freezing cannot restart warm-up
        # or change the method's order/model-evaluation count.
        if self.frozen is None and selected and len(history) == 3:
            self.frozen = [value.detach().clone() for value in history]
            self.frozen_at = iteration
        strength = self.options.history_strength * self.options.weight(iteration, self.steps)
        if self.frozen is None or len(history) != 3:
            strength = 0.0
        if strength == 0:
            return history, strength
        if strength == 1:
            return self.frozen, strength
        return [current + strength * (frozen - current)
                for current, frozen in zip(history, self.frozen)], strength


def sample_bent(model, x, sigmas, sampler_name, options, extra_args=None, callback=None):
    """Karras Euler/Heun updates with a private noise generator and pulse window.

    Equations follow k-diffusion 0.0.12 sampling.py (MIT). Only noise selection
    and the churn window differ. Disabled/zero-churn requests never enter here.
    Noise memory and the envelope are evaluated only at the selected pulses.
    """
    import torch
    from k_diffusion.sampling import to_d

    extra_args = extra_args or {}
    noise_source = NoiseMemory(x, options.seed, options.noise_memory)
    count = len(sigmas) - 1
    s_in = x.new_ones([x.shape[0]])
    for i in range(count):
        previous = x
        iteration = i + 1
        weight = options.weight(iteration, count)
        gamma = min(options.churn * weight / count, 2 ** 0.5 - 1)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            noise = noise_source.next()
            x = x + noise * options.noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if not torch.isfinite(x).all() or not torch.isfinite(denoised).all():
            raise ExperimentalSamplingStopped(previous, i)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat,
                      "denoised": denoised, "bend_applied": gamma > 0,
                      "bend_weight": weight, "noise_memory": options.noise_memory})
        dt = sigmas[i + 1] - sigma_hat
        if sampler_name == "euler" or sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            x = x + (d + d_2) / 2 * dt
        if not torch.isfinite(x).all():
            raise ExperimentalSamplingStopped(previous, i)
    return x
