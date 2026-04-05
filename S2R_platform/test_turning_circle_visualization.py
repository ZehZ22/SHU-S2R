import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from disturbances.current import decompose_current
from disturbances.wave import waveforce_irregular
from disturbances.wind import isherwood72
from ship_params import ShipParams
from vessels.kcs import KCS_ode, L, d_em, rho, U_des


def rk4(f, t, y, h, *args, **kwargs):
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args, **kwargs)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args, **kwargs)
    k4 = f(t + h, y + h * k3, *args, **kwargs)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def build_ext_force(wind_conf=None, wave_conf=None, ship=None, rng_seed=123):
    scale_f = 0.5 * rho * L * d_em * (U_des ** 2)
    scale_n = 0.5 * rho * (L ** 2) * d_em * (U_des ** 2)
    wave_rng = np.random.default_rng(rng_seed)
    wave_phase = wave_rng.uniform(0.0, 2.0 * np.pi, size=21)

    def _ext_force(t_nd, v, _):
        t_sec = t_nd * (L / U_des)
        psi = float(v[5])
        up = float(v[0])
        ship_speed = up * U_des

        x_si = 0.0
        y_si = 0.0
        n_si = 0.0

        if wind_conf is not None and ship is not None:
            wind_speed = float(wind_conf.get("V_wind", 0.0))
            wind_dir = math.radians(float(wind_conf.get("Psi_wind_deg", 0.0)))
            gamma_r = wind_dir - psi
            tau_wind, _, _, _ = isherwood72(
                gamma_r=gamma_r,
                V_r=wind_speed,
                Loa=ship.Loa,
                B=ship.B,
                ALw=ship.ALw,
                AFw=ship.AFw,
                A_SS=ship.A_SS,
                S=ship.S,
                C=ship.C,
                M=ship.M,
                rho_air=ship.rho_air,
            )
            x_si += float(tau_wind[0])
            y_si += float(tau_wind[1])
            n_si += float(tau_wind[2])

        if wave_conf is not None:
            wave_height = float(wave_conf.get("H", 0.0))
            wave_period = float(wave_conf.get("T", 10.0))
            wave_beta = math.radians(float(wave_conf.get("beta_deg", 0.0)))
            omega0 = 2.0 * np.pi / max(wave_period, 1e-6)
            omega = np.linspace(0.5 * omega0, 2.0 * omega0, wave_phase.size)
            beta_r = wave_beta - psi
            tau_wave = waveforce_irregular(
                t=t_sec,
                L=L,
                h=wave_height,
                T=wave_period,
                beta_r=beta_r,
                w=omega,
                fai=wave_phase,
                U=ship_speed,
            )
            x_si += float(tau_wave[0])
            y_si += float(tau_wave[1])
            n_si += float(tau_wave[2])

        return np.array([x_si / scale_f, y_si / scale_f, n_si / scale_n], dtype=float)

    return _ext_force


def make_current_func(speed_mps=0.0, beta_deg=0.0):
    def _current(t_nd, v, _):
        psi = float(v[5])
        beta_c = math.radians(beta_deg)
        current_nd = speed_mps / U_des
        return decompose_current(beta_c=beta_c, V_c=current_nd, psi=psi, U0=U_des)

    return _current


def simulate_turning_circle(
    delta_deg=35.0,
    dt=0.1,
    duration_nd=300.0,
    with_disturb=False,
    wind_conf=None,
    wave_conf=None,
    current_conf=None,
):
    ship = ShipParams()
    state = np.zeros(7, dtype=float)
    state[0] = 1.0

    ext_force = None
    current_func = None
    if with_disturb:
        ext_force = build_ext_force(wind_conf=wind_conf, wave_conf=wave_conf, ship=ship)
        current_func = make_current_func(
            speed_mps=float((current_conf or {}).get("Vc_mps", 0.0)),
            beta_deg=float((current_conf or {}).get("beta_c_deg", 0.0)),
        )

    steps = int(np.floor(duration_nd / dt)) + 1
    t_hist = np.zeros(steps, dtype=float)
    state_hist = np.zeros((steps, 7), dtype=float)
    delta_cmd = math.radians(delta_deg)

    t_nd = 0.0
    for i in range(steps):
        t_hist[i] = t_nd
        state_hist[i] = state
        state = rk4(
            KCS_ode,
            t_nd,
            state,
            dt,
            delta_cmd,
            ext_force=ext_force,
            ext_ctx=None,
            current_func=current_func,
            current_ctx=None,
        )
        t_nd += dt

    return t_hist, state_hist


def plot_turning_circle(t_hist, state_hist, units="nd", title_suffix="", save_path=None):
    x = state_hist[:, 3]
    y = state_hist[:, 4]
    psi_deg = np.degrees(state_hist[:, 5])
    delta_deg = np.degrees(state_hist[:, 6])
    yaw_rate_deg = np.degrees(state_hist[:, 2])
    speed_nd = state_hist[:, 0]

    if units == "m":
        x_plot = x * L
        y_plot = y * L
        xlabel = "x (m)"
        ylabel = "y (m)"
    else:
        x_plot = x
        y_plot = y
        xlabel = "x (nd)"
        ylabel = "y (nd)"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes.ravel()

    ax[0].plot(x_plot, y_plot, linewidth=2.0, color="tab:blue")
    ax[0].scatter([x_plot[0]], [y_plot[0]], color="tab:green", label="Start", zorder=3)
    ax[0].scatter([x_plot[-1]], [y_plot[-1]], color="tab:red", label="End", zorder=3)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(f"Turning Trajectory{title_suffix}")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t_hist, psi_deg, linewidth=2.0, color="tab:orange")
    ax[1].set_xlabel("t (nd)")
    ax[1].set_ylabel("Heading (deg)")
    ax[1].set_title("Heading Response")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t_hist, delta_deg, linewidth=2.0, color="tab:purple")
    ax[2].set_xlabel("t (nd)")
    ax[2].set_ylabel("Rudder (deg)")
    ax[2].set_title("Rudder Angle")
    ax[2].grid(True, alpha=0.3)

    ax[3].plot(t_hist, yaw_rate_deg, linewidth=1.8, color="tab:red", label="Yaw rate")
    ax[3].plot(t_hist, speed_nd, linewidth=1.8, color="tab:green", label="Surge speed")
    ax[3].set_xlabel("t (nd)")
    ax[3].set_ylabel("Response")
    ax[3].set_title("Yaw Rate and Surge Speed")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"figure_saved={save_path}")

    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
        return

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize KCS turning-circle test without loading a policy model.")
    parser.add_argument("--disturb", action="store_true", help="Enable wind, wave and current disturbances.")
    parser.add_argument("--delta-deg", type=float, default=35.0, help="Commanded rudder angle in degrees.")
    parser.add_argument("--dt", type=float, default=0.1, help="Nondimensional integration step.")
    parser.add_argument("--duration-nd", type=float, default=300.0, help="Simulation duration in nondimensional time.")
    parser.add_argument("--units", choices=["nd", "m"], default="nd", help="Trajectory plot units.")
    parser.add_argument("--wind-speed", type=float, default=12.0, help="Wind speed in m/s when disturbance is enabled.")
    parser.add_argument("--wind-dir-deg", type=float, default=45.0, help="Wind direction in degrees.")
    parser.add_argument("--wave-height", type=float, default=2.0, help="Wave height in meters.")
    parser.add_argument("--wave-period", type=float, default=8.0, help="Wave period in seconds.")
    parser.add_argument("--wave-dir-deg", type=float, default=135.0, help="Wave direction in degrees.")
    parser.add_argument("--current-speed", type=float, default=0.5, help="Current speed in m/s.")
    parser.add_argument("--current-dir-deg", type=float, default=155.0, help="Current direction in degrees.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    args = parser.parse_args()

    wind_conf = dict(V_wind=args.wind_speed, Psi_wind_deg=args.wind_dir_deg)
    wave_conf = dict(H=args.wave_height, T=args.wave_period, beta_deg=args.wave_dir_deg)
    current_conf = dict(Vc_mps=args.current_speed, beta_c_deg=args.current_dir_deg)

    t_hist, state_hist = simulate_turning_circle(
        delta_deg=args.delta_deg,
        dt=args.dt,
        duration_nd=args.duration_nd,
        with_disturb=args.disturb,
        wind_conf=wind_conf,
        wave_conf=wave_conf,
        current_conf=current_conf,
    )

    x_final = float(state_hist[-1, 3] * (L if args.units == "m" else 1.0))
    y_final = float(state_hist[-1, 4] * (L if args.units == "m" else 1.0))
    psi_final_deg = float(np.degrees(state_hist[-1, 5]))

    print(f"disturbance_enabled={args.disturb}")
    print(f"final_position=({x_final:.3f}, {y_final:.3f}) [{args.units}]")
    print(f"final_heading={psi_final_deg:.3f} deg")

    title_suffix = " (with disturbance)" if args.disturb else " (no disturbance)"
    plot_turning_circle(
        t_hist,
        state_hist,
        units=args.units,
        title_suffix=title_suffix,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
