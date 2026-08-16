"""树突 PSP 连续动力学的纯计算组件。

这里定义双指数 spike response、峰值和树突电位计算。函数不修改模型状态，
也不使用 RNG。strict reproduction 依赖当前公式和浮点累加顺序，结构重构
不得改成近似积分或重排求和。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


# Keep the standard-library implementation bound locally.  Long diagnostic
# runs can load third-party extensions and checkpoint state for many hours;
# using this private binding prevents an accidental module-attribute mutation
# from changing the PSP calculation halfway through a process.
_EXP = math.exp


def unscaled_kernel_peak(tau_m: float, tau_s: float) -> tuple[float, float]:
    """Return the peak time and value of the unscaled double exponential."""

    if not 0 < tau_s < tau_m:
        raise ValueError("time constants must satisfy 0 < tau_s < tau_m.")
    peak_time = (
        tau_m
        * tau_s
        / (tau_m - tau_s)
        * math.log(tau_m / tau_s)
    )
    peak = _EXP(-peak_time / tau_m) - _EXP(-peak_time / tau_s)
    return peak_time, peak


def kernel_peak_value(
    tau_m: float,
    tau_s: float,
    response_scale: float | None,
) -> float:
    """Return the exact peak after applying the configured response scale."""

    _peak_time, peak = unscaled_kernel_peak(tau_m, tau_s)
    return 1.0 if response_scale is None else response_scale * peak


def minimum_synchronous_synapses(
    *,
    weight: float,
    threshold: float,
    kernel_peak: float,
    voltage_tolerance: float = 0.0,
) -> int:
    """Minimum equal synchronous synapses needed to reach model threshold."""

    contribution = weight * kernel_peak
    if contribution <= 0.0:
        raise ValueError("weight and kernel peak must have a positive product")
    effective_threshold = threshold - voltage_tolerance
    return max(1, math.ceil(effective_threshold / contribution - 1e-12))


@dataclass(frozen=True)
class DSDynamicsParams:
    """Normalized DS-neuron parameters based on Eqs. (1)-(3) and Table I."""

    # PAPER GAP: LOCAL IMPLEMENTATION
    # Paper Eq.(1) names layer-specific tau_m/tau_s but publishes no numeric
    # values.  The normalized strict implementation uses 0.10 and 0.02.
    tau_m: float = 0.10
    tau_s: float = 0.02
    # PAPER GAP: LOCAL IMPLEMENTATION
    # Paper Eq.(1) exposes V0.  The public paper does not state unit-peak
    # normalization.  None means 1/unscaled_peak, not "no scale"; with current
    # taus effective V0 is about 1.87 and the PSP peak is exactly 1.
    response_scale: float | None = None
    v_rest: float = 0.0
    v_dep: float = 0.5
    depolarization_duration: float = 0.5
    oscillation_amplitude: float = 0.5
    oscillation_frequency: float = 1.0
    initial_phase: float = 0.0
    dendrite_threshold: float = 1.0
    soma_threshold: float = 1.0
    refractory_duration: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.tau_s < self.tau_m:
            raise ValueError("time constants must satisfy 0 < tau_s < tau_m.")
        if self.depolarization_duration <= 0 or self.refractory_duration < 0:
            raise ValueError("state durations must be nonnegative.")

    @property
    @lru_cache(maxsize=64)
    def kernel_peak_time(self) -> float:
        return unscaled_kernel_peak(self.tau_m, self.tau_s)[0]

    @property
    def kernel_scale(self) -> float:
        # PAPER GAP: LOCAL IMPLEMENTATION
        # Paper Eq.(1) exposes V0.  The public paper does not state unit-peak
        # normalization.  Strict response_scale=None means 1/unscaled_peak,
        # not "no scale"; current taus imply V0 about 1.87 and peak exactly 1.
        if self.response_scale is not None:
            return self.response_scale
        _peak_time, peak = unscaled_kernel_peak(self.tau_m, self.tau_s)
        return 1.0 / peak


@lru_cache(maxsize=4096)
def spike_response(elapsed: float, params: DSDynamicsParams) -> float:
    """Evaluate the double-exponential PSP term in paper Eq. (1).

    Inputs/outputs: elapsed time and dynamics parameters -> PSP voltage.
    State mutation: none.  Paper uncertainty: V0 normalization and both time
    constants are local strict choices, not published Zhang values.
    """

    if elapsed < 0:
        return 0.0
    # PAPER GAP: LOCAL IMPLEMENTATION
    # params.kernel_scale is V0 in Eq.(1); None has already become unit-peak
    # normalization rather than an omitted or paper-specified amplitude.
    return params.kernel_scale * (
        _EXP(-elapsed / params.tau_m) - _EXP(-elapsed / params.tau_s)
    )


@dataclass(frozen=True)
class DelayedSpike:
    firing_time: float
    delay: float
    weight: float

    @property
    def arrival_time(self) -> float:
        return self.firing_time + self.delay


def dendritic_potential(
    spikes: list[DelayedSpike],
    time: float,
    params: DSDynamicsParams,
) -> float:
    """Sum delayed distal PSPs into the dendritic voltage of paper Eq. (1).

    Inputs/outputs: delayed spikes and time -> voltage.  State mutation: none.
    Paper uncertainty: amplitude, time constants, and numerical crossing search
    are supplied by the local implementation.
    """

    return params.v_rest + sum(
        spike.weight * spike_response(time - spike.arrival_time, params)
        for spike in spikes
    )


@dataclass
class DSNeuronState:
    """Continuous state needed for depolarization, phase precession, and firing."""

    depolarized_at: float | None = None
    last_spike_time: float | None = None

    def trigger_dendritic_spike(self, time: float) -> None:
        self.depolarized_at = time

    def is_depolarized(self, time: float, params: DSDynamicsParams) -> bool:
        return (
            self.depolarized_at is not None
            and self.depolarized_at <= time
            and time - self.depolarized_at <= params.depolarization_duration
        )

    def is_refractory(self, time: float, params: DSDynamicsParams) -> bool:
        return (
            self.last_spike_time is not None
            and self.last_spike_time < time
            and time - self.last_spike_time < params.refractory_duration
        )

    def oscillation(self, time: float, params: DSDynamicsParams) -> float:
        if self.depolarized_at is None:
            return -params.oscillation_amplitude * math.sin(
                2.0 * math.pi * params.oscillation_frequency * time
                + params.initial_phase
            )
        elapsed = max(0.0, time - self.depolarized_at)
        # PAPER STATUS: PARTIAL
        # Paper describes phase precession and return to phi0, but not this
        # exact crossing-anchored -cos trajectory.  It is a local realization.
        return -params.oscillation_amplitude * math.cos(
            2.0 * math.pi * params.oscillation_frequency * elapsed
        )

    def membrane_potential(
        self,
        time: float,
        params: DSDynamicsParams,
        proximal: float = 0.0,
        apical: float = 0.0,
        inhibition: float = 0.0,
    ) -> float:
        """Evaluate the local soma approximation corresponding to paper Eq. (3).

        Inputs/outputs: time and zone voltages -> soma voltage.  State mutation:
        none.  Paper uncertainty: strict callers do not supply a continuous
        V_inh(t), and refractory suppression is stronger than the printed eta.
        """

        if self.is_refractory(time, params):
            # PAPER STATUS: PARTIAL
            # Paper uses eta=-theta for the rest of the current cycle.  Returning
            # -inf enforces suppression but is not the same voltage trajectory.
            return -math.inf
        distal = params.v_dep if self.is_depolarized(time, params) else 0.0
        # PAPER GAP: IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN
        # Eq.(3) explicitly contains V_inh(t).  This helper accepts inhibition,
        # but strict prediction currently calls it with the zero default.  The
        # complete trajectory is not reproduced, while its numerical form is
        # also absent from the public paper; exact author equivalence is unknown.
        return proximal + apical + distal + inhibition + self.oscillation(time, params)

    def try_fire(
        self,
        time: float,
        params: DSDynamicsParams,
        proximal: float = 0.0,
        apical: float = 0.0,
        inhibition: float = 0.0,
    ) -> bool:
        if self.membrane_potential(
            time, params, proximal=proximal, apical=apical, inhibition=inhibition
        ) < params.soma_threshold:
            return False
        self.last_spike_time = time
        self.depolarized_at = None
        return True
