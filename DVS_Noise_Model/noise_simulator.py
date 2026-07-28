"""Simple noise simulator for DVS threshold and refractory sweep."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dvs_noise_core import (
    Environment,
    EventCameraPixelSim,
    PixelParams,
    SimConfig,
    binned_binary_sequence,
    binned_symbol_sequence,
    default_illuminance_waveform,
    extract_events,
    match_events,
    summarize_event_stats,
    vp_distance_signed,
    _levenshtein_distance,
)


@dataclass
class NoiseParams:
    """Minimal noise simulator inputs."""
    theta: float = 0.01
    refractory_us: float = 200.0
    sigma_theta_static: float = 0.3


def noise_params_to_pixel(params: NoiseParams) -> PixelParams:
    """Convert theta + refractory into DVS_noise pixel parameters."""
    return PixelParams(
        C1=1.0,
        C2=1.0,
        theta=params.theta,
        sigma_theta_static=params.sigma_theta_static,
        refractory_s=params.refractory_us * 1e-6,
        I_leak1=0.0,
        I_leak2=0.0,
        I_leak_reset_mean=0.0,
        I_leak_reset_sigma=0.0,
    )


def simulate_noise_quality(
    theta: float,
    refractory_us: float,
    t_end: float = 0.05,
    dt: float = 5e-6,
    seed: int = 1,
    illuminance_fn: Optional[Callable] = None,
) -> dict:
    """Run the noise simulator and return simple noise quality metrics."""
    params = NoiseParams(theta=theta, refractory_us=refractory_us)
    env = Environment(T_kelvin=300.0, illuminance=100.0)
    pixel = noise_params_to_pixel(params)
    cfg = SimConfig(dt=dt, t_end=t_end, seed=seed)

    stim = illuminance_fn or (
        lambda tt: default_illuminance_waveform(tt, base=100.0, amp=80.0, f_hz=50.0)
    )

    noisy_sim = EventCameraPixelSim(env, pixel, cfg, illuminance_fn=stim)
    ideal_sim = EventCameraPixelSim(
        env,
        pixel,
        cfg,
        illuminance_fn=stim,
        qe_spatial=noisy_sim.qe_spatial,
    )

    noisy_trace = noisy_sim.run_trace(
        deterministic_photons=False,
        noiseless=False,
        ideal_reference=False,
    )
    ideal_trace = ideal_sim.run_trace(
        deterministic_photons=True,
        noiseless=True,
        ideal_reference=True,
    )

    t_noisy, p_noisy = extract_events(noisy_trace)
    t_ideal, p_ideal = extract_events(ideal_trace)
    duration = float(noisy_trace["t"][-1] - noisy_trace["t"][0] + cfg.dt)

    match = match_events(t_noisy, p_noisy, t_ideal, p_ideal, tol_s=20.0 * cfg.dt)
    precision = match["TP"] / (match["TP"] + match["FP"] + 1e-9)
    recall = match["TP"] / (match["TP"] + match["FN"] + 1e-9)
    snr = match["TP"] / (match["FP"] + match["FN"] + 1e-9)

    stats_ideal = summarize_event_stats(t_ideal, p_ideal, duration)
    n2 = 128 ** 2
    r_std_Meps = (stats_ideal["rate_hz"] * n2) / 1e6
    r_fp_Meps = (match["FP"] / duration * n2) / 1e6
    r_fn_Meps = (match["FN"] / duration * n2) / 1e6
    r_tp_Meps = (match["TP"] / duration * n2) / 1e6

    # Optional sequence similarity metrics for debugging
    start = float(noisy_trace["t"][0])
    end = float(noisy_trace["t"][-1] + cfg.dt)
    bin_s = 5.0 * cfg.dt
    sym_noisy = binned_symbol_sequence(t_noisy, p_noisy, start, end, bin_s)
    sym_ideal = binned_symbol_sequence(t_ideal, p_ideal, start, end, bin_s)
    lev_dist = _levenshtein_distance(sym_noisy, sym_ideal)
    vp = vp_distance_signed(t_noisy, p_noisy, t_ideal, p_ideal, q=4000.0)

    return {
        "theta": theta,
        "refractory_us": refractory_us,
        "TP": int(match["TP"]),
        "FP": int(match["FP"]),
        "FN": int(match["FN"]),
        "precision": precision,
        "recall": recall,
        "snr": snr,
        "snr_dB": 10.0 * np.log10(snr + 1e-9),
        "r_std_Meps": r_std_Meps,
        "r_fp_Meps": r_fp_Meps,
        "r_fn_Meps": r_fn_Meps,
        "r_tp_Meps": r_tp_Meps,
        "levenshtein": lev_dist,
        "vp_sum": vp["vp_sum"],
        "duration_s": duration,
    }
