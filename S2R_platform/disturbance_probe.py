import argparse
import numpy as np
import matplotlib.pyplot as plt

from vessels.kcs import L, U_des, d_em, rho
from disturbances.wind import isherwood72
from disturbances.wave import waveforce_irregular
from disturbances.current import decompose_current


def main():
    p = argparse.ArgumentParser(description='Probe wind/wave/current disturbance models over time')
    p.add_argument('--time-s', type=float, default=600.0, help='Duration in seconds')
    p.add_argument('--dt-s', type=float, default=1.0, help='Time step in seconds')
    p.add_argument('--psi-deg', type=float, default=0.0, help='Ship heading (deg) assumed constant')

    # Wind
    p.add_argument('--wind-speed', type=float, default=1.0, help='Wind speed (m/s)')
    p.add_argument('--wind-dir', type=float, default=45.0, help='Wind direction (deg from x)')

    # Wave
    p.add_argument('--wave-H', type=float, default=1.0, help='Significant wave height H (m)')
    p.add_argument('--wave-T', type=float, default=8.0, help='Wave period T (s)')
    p.add_argument('--wave-dir', type=float, default=45.0, help='Wave direction (deg from x)')

    # Current
    p.add_argument('--current-speed', type=float, default=1.0, help='Current speed (m/s)')
    p.add_argument('--current-dir', type=float, default=60.0, help='Current direction (deg from x)')

    args = p.parse_args()

    n = int(np.floor(args.time_s / args.dt_s)) + 1
    t = np.linspace(0.0, args.time_s, n)
    psi = np.radians(args.psi_deg) * np.ones_like(t)

    # Storage
    wind_X = np.zeros_like(t)
    wind_Y = np.zeros_like(t)
    wind_N = np.zeros_like(t)

    wave_X = np.zeros_like(t)
    wave_Y = np.zeros_like(t)
    wave_N = np.zeros_like(t)

    uc = np.zeros_like(t)
    vc = np.zeros_like(t)

    # Precompute wave spectral components
    w0 = 2 * np.pi / max(args.wave_T, 1e-6)
    w = np.linspace(0.8 * w0, 1.2 * w0, 21)
    rng = np.random.default_rng(123)
    fai = rng.uniform(0, 2 * np.pi, size=w.size)

    for i in range(n):
        # Wind (SI)
        gamma_r = np.radians(args.wind_dir) - psi[i]
        tauW, *_ = isherwood72(
            gamma_r=gamma_r,
            V_r=args.wind_speed,
            Loa=L,
            B=32.2,
            ALw=L * d_em / 2.0,
            AFw=L * d_em / 6.0,
            A_SS=0.0,
            S=L,
            C=0.2 * L,
            M=1,
            rho_air=1.225,
        )
        wind_X[i], wind_Y[i], wind_N[i] = tauW

        # Wave (SI)
        beta_r = np.radians(args.wave_dir) - psi[i]
        tau_wave = waveforce_irregular(
            t=t[i], L=L, h=args.wave_H, T=args.wave_T, beta_r=beta_r, w=w, fai=fai, U=0.0,
        )
        wave_X[i], wave_Y[i], wave_N[i] = tau_wave

        # Current (nondimensional body components)
        Vc_nd = args.current_speed / U_des
        u_c, v_c = decompose_current(np.radians(args.current_dir), Vc_nd, psi[i], U_des)
        uc[i], vc[i] = u_c, v_c

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax = axes[0]
    ax.plot(t, wind_X, label='Wind X (N)')
    ax.plot(t, wind_Y, label='Wind Y (N)')
    ax.plot(t, wind_N, label='Wind N (N·m)')
    ax.set_ylabel('Wind loads')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(t, wave_X, label='Wave X (N)')
    ax.plot(t, wave_Y, label='Wave Y (N)')
    ax.plot(t, wave_N, label='Wave N (N·m)')
    ax.set_ylabel('Wave loads')
    ax.legend()
    ax.grid(True)

    # Convert current components to m/s for display
    uc_mps = uc * U_des
    vc_mps = vc * U_des

    ax = axes[2]
    ax.plot(t, uc_mps, label='u_c (m/s)')
    ax.plot(t, vc_mps, label='v_c (m/s)')
    ax.set_ylabel('Current (m/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
