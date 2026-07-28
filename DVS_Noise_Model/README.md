# DVS Noise Heatmap Simplified

This folder contains a minimal self-contained implementation for generating a threshold vs refractory-period noise heatmap.

## Files

- `dse.py`
  - The main entry point.
  - Sweeps `theta` (threshold) and `refractory_us` (refractory period).
  - Calls `noise_simulator.simulate_noise_quality(...)` for each point.
  - Plots a heatmap of `snr_dB` and saves it as `dse_snr_heatmap.pdf`.

- `noise_simulator.py`
  - The noise simulator module.
  - Uses a simplified DVS pixel model to generate noisy and ideal event traces.
  - Computes event matching metrics: `TP`, `FP`, `FN`.
  - Calculates `snr_dB` from those metrics.

- `dvs_noise_core.py`
  - Contains the minimal DVS noise model core used by the simulator.
  - Defines `Environment`, `PixelParams`, `SimConfig`, and `EventCameraPixelSim`.
  - Includes event extraction and matching helpers.

## How it works

1. `dse.py` defines a sweep over threshold and refractory period.
2. For each sweep point, `simulate_noise_quality(...)` runs two simulations:
   - a noisy DVS pixel trace
   - an ideal reference trace
3. It matches events between the noisy output and ideal reference.
4. It computes `snr_dB = 10 * log10(TP / (FP + FN))`.
5. `dse.py` then plots the resulting `snr_dB` values on a 2D heatmap.

## Run

From the folder containing `dse.py`:

```bash
python3 dse.py
```

The output file will be:

- `dse_snr_heatmap.pdf`

## Notes

- This folder is intentionally self-contained.
- It does not require the original `DVS_noise` package or the larger power DSE code.
