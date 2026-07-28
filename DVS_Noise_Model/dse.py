"""Threshold-refractory period noise tradeoff DSE."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise_simulator import simulate_noise_quality


@dataclass
class DSEConfig:
    theta_range: Tuple[float, float, int] = (0.007, 0.016, 10)
    refractory_us_range: Tuple[float, float, int] = (140.0, 300.0, 10)


def sweep_noise(
    config: DSEConfig,
    t_end: float = 0.02,
    dt: float = 5e-6,
    seed: int = 1,
    illuminance_fn: Optional[Callable] = None,
) -> List[Dict[str, float]]:
    """Sweep theta + refractory and return only noise metrics."""
    theta_vals = np.linspace(*config.theta_range)
    refractory_vals = np.linspace(*config.refractory_us_range)
    results = []
    count = 0

    for theta in theta_vals:
        for refractory_us in refractory_vals:
            noise_metrics = simulate_noise_quality(
                theta=theta,
                refractory_us=refractory_us,
                t_end=t_end,
                dt=dt,
                seed=seed + count,
                illuminance_fn=illuminance_fn,
            )

            result = {
                **noise_metrics,
                "theta": theta,
                "refractory_us": refractory_us,
            }
            results.append(result)
            count += 1

    return results


def plot_heatmap(
    results: List[Dict[str, float]],
    x_key: str = "theta",
    y_key: str = "refractory_us",
    z_key: str = "snr_dB",
    x_label: str = "Threshold (mV)",
    y_label: str = "Refractory Period (µs)",
    z_label: str = "SNR (dB)",
    x_scale: float = 1000.0,
    filename: str = "dse_snr_heatmap.pdf",
) -> None:
    """Plot a simple heatmap of one metric over two sweep axes."""
    x_vals = np.unique([r[x_key] for r in results])
    y_vals = np.unique([r[y_key] for r in results])
    Z = np.full((len(y_vals), len(x_vals)), np.nan)

    for r in results:
        i = int(np.where(y_vals == r[y_key])[0][0])
        j = int(np.where(x_vals == r[x_key])[0][0])
        Z[i, j] = r[z_key]

    x_edges = np.concatenate([x_vals - np.diff(x_vals, prepend=x_vals[0]) / 2, [x_vals[-1] + (x_vals[-1] - x_vals[-2]) / 2]])
    y_edges = np.concatenate([y_vals - np.diff(y_vals, prepend=y_vals[0]) / 2, [y_vals[-1] + (y_vals[-1] - y_vals[-2]) / 2]])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(x_edges, y_edges, Z, cmap="viridis", shading="flat")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * x_scale:.1f}")
    cbar = plt.colorbar(im, ax=ax, label=z_label)
    cbar.ax.set_ylabel(z_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run a theta/refractory sweep and save the SNR heatmap."""
    cfg = DSEConfig()
    results = sweep_noise(cfg)
    plot_heatmap(results, filename="dse_snr_heatmap.pdf")


if __name__ == "__main__":
    main()
