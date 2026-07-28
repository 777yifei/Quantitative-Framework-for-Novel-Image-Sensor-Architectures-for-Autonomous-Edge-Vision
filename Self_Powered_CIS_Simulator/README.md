# Self-Powered CIS Simulator

This folder contains a compact self-contained simulator for exploring a self-powered CIS design space.

## Files

- `Self-Powered_CIS_Simulator.py`
  - The main entry point.
  - Sweeps over design parameters such as total system weight, resolution, and sampling rate.
  - Computes frame time and the FPS improvement ratio relative to a baseline.

## How it works

1. The script evaluates whether the system can sustain a given frame rate by comparing harvested power with the power required for CIS operation.
2. It computes the frame period $T$ from a power-balance equation, where the energy consumed per frame is balanced by the available harvested energy over the same time.
3. From this frame period, it derives the achievable FPS and compares it with a baseline to obtain the improvement ratio.
4. The results are collected into a design-space table and visualized as 3D scatter plots.

## Run

```bash
python3 Self-Powered_CIS_Simulator.py
```

The output file will be:

- `design_space.pdf`

## Notes

- This script is intentionally self-contained.
- It does not depend on the other project modules and only needs standard Python packages such as NumPy, pandas, and matplotlib.
