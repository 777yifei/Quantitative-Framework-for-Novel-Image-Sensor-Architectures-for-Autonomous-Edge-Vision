#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D scatter

# Physical constants used by the model
CIS_density = 135        # kg/m^2
E_p = 6.3e-9             # J per pixel operation
sp_density = 10          # kg/m^2
E_light = 486            # W/m^2
eta_sp = 0.07            # solar panel efficiency
T_IM = 0.01              # s minimum integration time


def params_from_FF(FF, area_pixel_base, eta_sensor_org):
    """Convert fill factor into effective pixel area and sensor efficiency."""
    scale = 2.0 - FF
    if scale <= 0:
        raise ValueError("2 - FF must be positive")
    area_pixel_ours = area_pixel_base * scale
    eta_sensor_adj = eta_sensor_org / scale
    return area_pixel_ours, eta_sensor_adj


def calculate_T(total_weight, k1, R, area_pixel, eta_sensor, use_P_CIS=True):
    """Compute the frame time T for a design point."""
    Area_sp = (total_weight - R * area_pixel * CIS_density) / sp_density
    Area_sp = np.maximum(Area_sp, 0.0)

    P_CIS = (E_light * eta_sensor * area_pixel) if use_P_CIS else 0.0
    P_SP = E_light * eta_sp * Area_sp

    num = k1 * E_p + k1 * T_IM * P_CIS
    den = R * P_CIS + P_SP

    with np.errstate(divide='ignore', invalid='ignore'):
        T = num / den
    return np.maximum(T, T_IM)


def make_dataframe(N=100, tw_max=0.033,
                   area_pixel_ours=4.5e-11,
                   area_pixel_base=4.5e-11,
                   eta_sensor=0.03):
    """Build the design-space table of weights, resolutions, and frame metrics."""
    tw_vals = np.linspace(0.0, tw_max, N)
    Rn_vals = np.linspace(0.0, 0.98, N)
    kn_vals = np.linspace(0.0, 1.0, N)
    TW, Rn, Kn = np.meshgrid(tw_vals, Rn_vals, kn_vals, indexing='ij')

    R_max = TW / (CIS_density * area_pixel_ours)
    R_act = Rn * R_max
    k1_act = Kn * R_act

    T_ours = calculate_T(TW, k1_act, R_act, area_pixel_ours, eta_sensor, True)
    T_base = calculate_T(TW, k1_act, R_act, area_pixel_base, eta_sensor, False)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = T_base / T_ours

    mask = np.isfinite(ratio)
    return pd.DataFrame({
        'total_weight': TW[mask],
        'R_actual': R_act[mask],
        'k_norm': Kn[mask],
        'R_max': R_max[mask],
        'FPS_ratio': ratio[mask],
        'T_ours': T_ours[mask],
        'T_base': T_base[mask],
    })


def main():
    FF = 0.96
    area_base = 8e-11
    eta_sensor_org = 0.04

    area_ours, eta_sensor_adj = params_from_FF(FF, area_base, eta_sensor_org)
    print(f"[INFO] FF={FF:.2f} -> area_ours={area_ours:.3e}, eta_sensor={eta_sensor_adj:.3f}")

    df = make_dataframe(N=100, tw_max=0.033,
                        area_pixel_ours=area_ours,
                        area_pixel_base=area_base,
                        eta_sensor=eta_sensor_adj)

    sample_df = df.sample(frac=0.02, random_state=42)
    x = sample_df['total_weight']
    y = sample_df['R_actual']
    z = sample_df['k_norm']

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(x, y, z, c=sample_df['FPS_ratio'], cmap='viridis_r', s=5, alpha=0.4)
    ax1.set_title('FPS Improvement Ratio')
    ax1.set_xlabel('Total Weight (kg)')
    ax1.set_ylabel('R (pixels)')
    ax1.set_zlabel('Sampling Rate')
    ax1.view_init(elev=30, azim=150)
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('FPS Ratio')

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(x, y, z, c=sample_df['T_ours'], cmap='plasma_r', s=5, alpha=0.4)
    ax2.set_title('T_ours Distribution')
    ax2.set_xlabel('Total Weight (kg)')
    ax2.set_ylabel('R (pixels)')
    ax2.set_zlabel('Sampling Rate')
    ax2.view_init(elev=30, azim=150)
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Frame Time of Self-powered Design (s)')

    plt.tight_layout(pad=0.5)
    plt.savefig('design_space.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()


if __name__ == '__main__':
    main()
