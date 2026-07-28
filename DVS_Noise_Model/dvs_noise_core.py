"""Minimal DVS noise model core used by the simplified DSE scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


@dataclass
class Environment:
    """External conditions for the DVS pixel simulation."""
    T_kelvin: float
    illuminance: float


@dataclass
class PixelParams:
    photons_per_lux_per_s: float = 2e5
    qe_mean: float = 0.3
    qe_sigma: float = 0.02
    idc_ref: float = 50.0
    T_ref: float = 300.0
    Ea_over_k: float = 4000.0
    sigma_N_dc: float = 0.0
    vlog_gain: float = 0.2
    i0: float = 1.0
    sigma_G_log: float = 0.0
    R_LPF: float = 159e3
    C_PD: float = 1e-9
    R_HPF: float = 31.8e-3
    C1: float = 1.0
    C2: float = 1.0
    I_leak1: float = 0.0
    I_leak2: float = 0.0
    I_leak_reset_mean: float = 0.5
    I_leak_reset_sigma: float = 0.5
    vreset_mean: float = 0.0
    sigma_vreset: float = 1e-3
    theta: float = 0.01
    sigma_theta_static: float = 0.3
    sigma_theta_temp: float = 0.0
    refractory_s: float = 200e-6
    sigma_log_v: float = 1e-3
    sigma_vdiff_white: float = 0.0
    use_signed_events: bool = True
    cmp_pulse_s: float = 2e-5
    event_pulse_s: float = 2e-5


@dataclass
class SimConfig:
    dt: float = 5e-6
    t_end: float = 0.05
    seed: int = 1


def default_illuminance_waveform(
    t: np.ndarray, base: float = 100.0, amp: float = 80.0, f_hz: float = 50.0
) -> np.ndarray:
    return base + amp * np.sin(2.0 * np.pi * f_hz * t)


class EventCameraPixelSim:
    """Minimal DVS pixel simulator used by the noise sweep."""

    def __init__(
        self,
        env: Environment,
        p: PixelParams,
        cfg: SimConfig,
        illuminance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        qe_spatial: Optional[float] = None,
    ):
        self.env = env
        self.p = p
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        if qe_spatial is None:
            qe_gain = self.rng.normal(loc=1.0, scale=p.qe_sigma)
            qe_gain = max(0.0, qe_gain)
            self.qe_spatial = p.qe_mean * qe_gain
        else:
            self.qe_spatial = float(qe_spatial)

        self.illum_fn = illuminance_fn if illuminance_fn is not None else (
            lambda tt: default_illuminance_waveform(tt)
        )

        tau_lpf = p.R_LPF * p.C_PD
        tau_hpf = p.R_HPF * p.C1
        self.fc_lpf_hz = 1.0 / (2.0 * np.pi * tau_lpf) if tau_lpf > 0 else 0.0
        self.fc_hpf_hz = 1.0 / (2.0 * np.pi * tau_hpf) if tau_hpf > 0 else 0.0

        self.alpha_lpf = self._alpha_from_fc(self.fc_lpf_hz, cfg.dt)
        self.alpha_hpf = self._alpha_from_fc(self.fc_hpf_hz, cfg.dt, for_hpf=True)

        sigma_abs = p.sigma_theta_static * p.theta if p.sigma_theta_static > 0 else 0.0
        self.theta_static = float(self.rng.normal(0.0, sigma_abs)) if sigma_abs > 0 else 0.0

    @staticmethod
    def _alpha_from_fc(fc_hz: float, dt: float, for_hpf: bool = False) -> float:
        if fc_hz is None or fc_hz <= 0.0:
            return 0.0
        tau = 1.0 / (2.0 * np.pi * fc_hz)
        if for_hpf:
            return float(tau / (tau + dt))
        return float(np.exp(-dt / tau))

    def _dark_rate(self, T: float, noiseless: bool = False) -> float:
        base = self.p.idc_ref * float(np.exp(self.p.Ea_over_k * (1.0 / self.p.T_ref - 1.0 / T)))
        if (not noiseless) and self.p.sigma_N_dc > 0:
            base += float(self.rng.normal(0.0, self.p.sigma_N_dc))
        return max(0.0, base)

    def _mean_e_rate(self, L: float, noiseless: bool = False) -> float:
        photon_rate = max(0.0, L) * self.p.photons_per_lux_per_s
        signal = self.qe_spatial * photon_rate
        dark = self._dark_rate(self.env.T_kelvin, noiseless)
        return signal + dark

    def _poisson_count(self, mean_rate: float, dt: float) -> int:
        lam = max(0.0, mean_rate) * dt
        return int(self.rng.poisson(lam=lam))

    @staticmethod
    def _rate_from_count(cnt: int, dt: float) -> float:
        return float(cnt) / dt if dt > 0 else 0.0

    def _log_from_rate(self, rate: float, noiseless: bool = False) -> float:
        r = rate + self.p.i0
        v = self.p.vlog_gain * float(np.log(r))
        if (not noiseless) and self.p.sigma_G_log > 0:
            v *= 1.0 + float(self.rng.normal(0.0, self.p.sigma_G_log))
        return v

    def _lpf(self, y_prev: float, x: float) -> float:
        if self.fc_lpf_hz <= 0.0:
            return x
        return self.alpha_lpf * y_prev + (1.0 - self.alpha_lpf) * x

    def _hpf(self, y_prev: float, x: float, x_prev: float) -> float:
        if self.fc_hpf_hz <= 0.0:
            return x
        return self.alpha_hpf * (y_prev + x - x_prev)

    def _theta_eff(self) -> float:
        temp = 0.0
        if self.p.sigma_theta_temp > 0:
            temp = float(self.rng.normal(0.0, self.p.sigma_theta_temp * self.p.theta))
        return float(self.p.theta + self.theta_static + temp)

    def _sample_vreset(self) -> float:
        v = float(self.p.vreset_mean)
        if self.p.sigma_vreset > 0.0:
            v += float(self.rng.normal(0.0, self.p.sigma_vreset))
        return v

    def _can_fire(self, tk: float, last_event_t: float) -> bool:
        if self.p.refractory_s is None or self.p.refractory_s <= 0.0:
            return True
        return (tk - last_event_t) >= self.p.refractory_s

    def run_trace(
        self,
        deterministic_photons: bool = False,
        noiseless: bool = False,
        ideal_reference: bool = False,
    ) -> Dict[str, np.ndarray]:
        dt = self.cfg.dt
        n = int(np.ceil(self.cfg.t_end / dt))
        t = np.arange(n) * dt
        L_t = np.maximum(self.illum_fn(t), 0.0)

        vlog_pre = np.zeros(n)
        vlog_lpf = np.zeros(n)
        vlog_hpf = np.zeros(n)
        vref = np.zeros(n)
        vdiff = np.zeros(n)

        on_lvl = np.zeros(n, dtype=int)
        off_lvl = np.zeros(n, dtype=int)
        on_edge = np.zeros(n, dtype=int)
        off_edge = np.zeros(n, dtype=int)
        on_pulse = np.zeros(n, dtype=int)
        off_pulse = np.zeros(n, dtype=int)
        event_pulse = np.zeros(n, dtype=int)
        event_pol = np.zeros(n, dtype=int)
        theta_eff_arr = np.zeros(n)

        vlog_lpf_state = 0.0
        hpf_state = 0.0
        vlog_lpf_prev = 0.0
        vref_state = 0.0
        leak_acc = 0.0
        last_event_time = -np.inf
        pending = False
        pending_pol = 0

        cmp_w = max(1, int(np.round(self.p.cmp_pulse_s / dt)))
        evt_w = max(1, int(np.round(self.p.event_pulse_s / dt)))
        cmp_on_rem = 0
        cmp_off_rem = 0
        evt_rem = 0

        mean0 = self._mean_e_rate(float(L_t[0]), noiseless)
        vlog0 = self._log_from_rate(mean0, noiseless)
        vlog_lpf_state = vlog0
        vlog_lpf_prev = vlog_lpf_state
        hpf_state = 0.0
        vref_state = 0.0

        prev_on_lvl = 0
        prev_off_lvl = 0

        for k in range(n):
            tk = float(t[k])
            mean_rate = self._mean_e_rate(float(L_t[k]), noiseless)
            if deterministic_photons:
                rate_est = mean_rate
            else:
                cnt = self._poisson_count(mean_rate, dt)
                rate_est = self._rate_from_count(cnt, dt)

            v = self._log_from_rate(rate_est, noiseless)
            if (not noiseless) and (self.p.sigma_log_v > 0):
                v += float(self.rng.normal(0.0, self.p.sigma_log_v))
            vlog_pre[k] = v

            vlog_lpf_state = self._lpf(vlog_lpf_state, v)
            vlog_lpf[k] = vlog_lpf_state
            hpf_state = self._hpf(hpf_state, vlog_lpf_state, vlog_lpf_prev)
            vlog_hpf[k] = hpf_state
            vlog_lpf_prev = vlog_lpf_state

            if ideal_reference:
                ileak = 0.0
            else:
                ileak = self.p.I_leak1 + self.p.I_leak2 + self.p.I_leak_reset_mean
                if (not noiseless) and self.p.I_leak_reset_sigma > 0:
                    ileak += float(self.rng.normal(0.0, self.p.I_leak_reset_sigma))
            leak_acc += ileak * dt

            vdiff_k = (self.p.C1 * (hpf_state - vref_state) - leak_acc) / self.p.C2
            if (not noiseless) and (self.p.sigma_vdiff_white > 0):
                vdiff_k += float(self.rng.normal(0.0, self.p.sigma_vdiff_white))
            vdiff[k] = vdiff_k

            theta_eff = float(self.p.theta) if noiseless else self._theta_eff()
            theta_eff_arr[k] = theta_eff

            if self.p.use_signed_events:
                on_now = 1 if (vdiff_k >= +theta_eff) else 0
                off_now = 1 if (vdiff_k <= -theta_eff) else 0
            else:
                on_now = 1 if (vdiff_k >= +theta_eff) else 0
                off_now = 0

            on_lvl[k] = on_now
            off_lvl[k] = off_now

            on_edge_k = 1 if (on_now == 1 and prev_on_lvl == 0) else 0
            off_edge_k = 1 if (off_now == 1 and prev_off_lvl == 0) else 0
            on_edge[k] = on_edge_k
            off_edge[k] = off_edge_k
            prev_on_lvl = on_now
            prev_off_lvl = off_now

            if on_edge_k == 1:
                cmp_on_rem = cmp_w
            if off_edge_k == 1:
                cmp_off_rem = cmp_w

            on_pulse[k] = 1 if cmp_on_rem > 0 else 0
            off_pulse[k] = 1 if cmp_off_rem > 0 else 0
            cmp_on_rem = max(0, cmp_on_rem - 1)
            cmp_off_rem = max(0, cmp_off_rem - 1)

            req = (on_edge_k == 1) or (off_edge_k == 1)
            req_pol = +1 if on_edge_k == 1 else (-1 if off_edge_k == 1 else 0)
            can_fire = self._can_fire(tk, last_event_time)

            if pending and can_fire:
                still_trig = (vdiff_k >= +theta_eff) if pending_pol == +1 else (vdiff_k <= -theta_eff)
                if still_trig:
                    last_event_time = tk
                    evt_rem = evt_w
                    event_pol[k] = pending_pol
                    vref_state = hpf_state + (0.0 if noiseless else self._sample_vreset())
                    leak_acc = 0.0
                pending = False
                pending_pol = 0

            if req:
                if can_fire:
                    last_event_time = tk
                    evt_rem = evt_w
                    event_pol[k] = req_pol
                    vref_state = hpf_state + (0.0 if noiseless else self._sample_vreset())
                    leak_acc = 0.0
                else:
                    pending = True
                    pending_pol = req_pol

            event_pulse[k] = 1 if evt_rem > 0 else 0
            evt_rem = max(0, evt_rem - 1)
            vref[k] = vref_state

        return {
            "t": t,
            "L_t": L_t,
            "vlog_pre": vlog_pre,
            "vlog_lpf": vlog_lpf,
            "vlog_hpf": vlog_hpf,
            "vref": vref,
            "vdiff": vdiff,
            "theta_eff": theta_eff_arr,
            "on_lvl": on_lvl,
            "off_lvl": off_lvl,
            "on_edge": on_edge,
            "off_edge": off_edge,
            "on_pulse": on_pulse,
            "off_pulse": off_pulse,
            "event_pulse": event_pulse,
            "event_pol": event_pol,
        }


def extract_events(tr: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    t = tr["t"]
    pol = tr["event_pol"]
    idx = np.nonzero(pol != 0)[0]
    return t[idx], pol[idx].astype(int)


def _match_events_hungarian(
    dt_mat: np.ndarray, valid: np.ndarray, n_n: int, n_i: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return -1, np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    INVALID = 1e12
    cost = np.where(valid, np.abs(dt_mat), INVALID)
    row_ind, col_ind = linear_sum_assignment(cost)

    timing_err = []
    noisy_idx = []
    ideal_idx = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < INVALID:
            timing_err.append(float(dt_mat[r, c]))
            noisy_idx.append(r)
            ideal_idx.append(c)
    return len(timing_err), np.array(timing_err, dtype=float), np.array(noisy_idx), np.array(ideal_idx)


def _match_events_greedy(
    dt_mat: np.ndarray, valid: np.ndarray, n_n: int, n_i: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    pairs = []
    for i in range(n_n):
        for j in range(n_i):
            if valid[i, j]:
                pairs.append((np.abs(dt_mat[i, j]), i, j, dt_mat[i, j]))
    pairs.sort(key=lambda x: x[0])

    matched_noisy = np.zeros(n_n, dtype=bool)
    matched_ideal = np.zeros(n_i, dtype=bool)
    timing_err = []
    noisy_idx = []
    ideal_idx = []
    for _abs_dt, i, j, dt in pairs:
        if matched_noisy[i] or matched_ideal[j]:
            continue
        matched_noisy[i] = True
        matched_ideal[j] = True
        timing_err.append(float(dt))
        noisy_idx.append(i)
        ideal_idx.append(j)
    return len(timing_err), np.array(timing_err, dtype=float), np.array(noisy_idx), np.array(ideal_idx)


def match_events(
    t_noisy: np.ndarray,
    pol_noisy: np.ndarray,
    t_ideal: np.ndarray,
    pol_ideal: np.ndarray,
    tol_s: float,
) -> Dict[str, object]:
    if len(t_noisy) == 0 and len(t_ideal) == 0:
        return {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "timing_err": np.array([], dtype=float),
            "matched_noisy_idx": np.array([], dtype=int),
            "matched_ideal_idx": np.array([], dtype=int),
        }

    n_n = len(t_noisy)
    n_i = len(t_ideal)
    dt_mat = np.subtract.outer(t_noisy, t_ideal)
    pol_match = np.outer(pol_noisy, pol_ideal) > 0
    in_tol = np.abs(dt_mat) <= tol_s
    valid = pol_match & in_tol

    TP, timing_err, noisy_idx, ideal_idx = _match_events_hungarian(dt_mat, valid, n_n, n_i)
    if TP < 0:
        TP, timing_err, noisy_idx, ideal_idx = _match_events_greedy(dt_mat, valid, n_n, n_i)

    FP = n_n - TP
    FN = n_i - TP
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "timing_err": timing_err,
        "matched_noisy_idx": noisy_idx,
        "matched_ideal_idx": ideal_idx,
    }


def summarize_event_stats(t_evt: np.ndarray, pol_evt: np.ndarray, T: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(t_evt)
    out["count"] = float(n)
    out["rate_hz"] = float(n / T) if T > 0 else 0.0
    n_on = int(np.sum(pol_evt > 0))
    n_off = int(np.sum(pol_evt < 0))
    out["on_frac"] = float(n_on / n) if n > 0 else 0.0
    out["off_frac"] = float(n_off / n) if n > 0 else 0.0
    if n >= 2:
        isi = np.diff(t_evt)
        mu = float(np.mean(isi))
        sd = float(np.std(isi))
        out["isi_mean_s"] = mu
        out["isi_std_s"] = sd
        out["isi_cv"] = float(sd / mu) if mu > 0 else 0.0
    else:
        out["isi_mean_s"] = float("nan")
        out["isi_std_s"] = float("nan")
        out["isi_cv"] = float("nan")
    return out


def _levenshtein_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.size == 0:
        return int(b.size)
    if b.size == 0:
        return int(a.size)
    if b.size > a.size:
        a, b = b, a

    prev = np.arange(b.size + 1, dtype=int)
    curr = np.zeros(b.size + 1, dtype=int)
    for i in range(1, a.size + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, b.size + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return int(prev[-1])


def binned_symbol_sequence(
    t_evt: np.ndarray,
    pol_evt: np.ndarray,
    t_start: float,
    t_end: float,
    bin_s: float,
) -> np.ndarray:
    n_bins = int(np.ceil((t_end - t_start) / bin_s))
    sym = np.zeros(n_bins, dtype=int)
    if t_evt.size == 0:
        return sym
    idx = np.floor((t_evt - t_start) / bin_s).astype(int)
    valid = (idx >= 0) & (idx < n_bins)
    idx = idx[valid]
    pol = pol_evt[valid]
    on_present = np.zeros(n_bins, dtype=bool)
    off_present = np.zeros(n_bins, dtype=bool)
    on_present[idx[pol > 0]] = True
    off_present[idx[pol < 0]] = True
    sym[on_present & ~off_present] = 1
    sym[~on_present & off_present] = 2
    sym[on_present & off_present] = 3
    return sym


def binned_binary_sequence(
    t_evt: np.ndarray,
    pol_evt: np.ndarray,
    which_pol: int,
    t_start: float,
    t_end: float,
    bin_s: float,
) -> np.ndarray:
    n_bins = int(np.ceil((t_end - t_start) / bin_s))
    seq = np.zeros(n_bins, dtype=int)
    if t_evt.size == 0:
        return seq
    mask = (pol_evt > 0) if which_pol > 0 else (pol_evt < 0)
    tt = t_evt[mask]
    if tt.size == 0:
        return seq
    idx = np.floor((tt - t_start) / bin_s).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    seq[idx] = 1
    return seq


def victor_purpura_distance(t1: np.ndarray, t2: np.ndarray, q: float) -> float:
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    n = t1.size
    m = t2.size
    D = np.zeros((n + 1, m + 1), dtype=float)
    D[:, 0] = np.arange(n + 1, dtype=float)
    D[0, :] = np.arange(m + 1, dtype=float)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_move = q * abs(t1[i - 1] - t2[j - 1])
            D[i, j] = min(
                D[i - 1, j] + 1.0,
                D[i, j - 1] + 1.0,
                D[i - 1, j - 1] + cost_move,
            )
    return float(D[n, m])


def vp_distance_signed(
    t_evt: np.ndarray,
    pol_evt: np.ndarray,
    t_ide: np.ndarray,
    pol_ide: np.ndarray,
    q: float,
) -> Dict[str, float]:
    t_on_evt = t_evt[pol_evt > 0]
    t_off_evt = t_evt[pol_evt < 0]
    t_on_ide = t_ide[pol_ide > 0]
    t_off_ide = t_ide[pol_ide < 0]
    d_on = victor_purpura_distance(t_on_evt, t_on_ide, q=q)
    d_off = victor_purpura_distance(t_off_evt, t_off_ide, q=q)
    return {"vp_on": float(d_on), "vp_off": float(d_off), "vp_sum": float(d_on + d_off)}
