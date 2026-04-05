import argparse
import numpy as np

from disturbances.current import decompose_current
from disturbances.wave import waveforce_irregular
from disturbances.wind import isherwood72
from ship_params import ShipParams
from vessels.kcs import L as L_ship, U_des
from vessels.mariner import mariner


def rk4(f, t, y, h, *args, **kwargs):
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args, **kwargs)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args, **kwargs)
    k4 = f(t + h, y + h * k3, *args, **kwargs)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_wind_force_func(wind_conf=None, ship=None):
    def _wind_force(_, v):
        if wind_conf is None or ship is None:
            return None

        v_wind = wind_conf.get("V_wind", 0.0)
        psi_wind = np.radians(wind_conf.get("Psi_wind_deg", 0.0))
        psi = v[5]

        # mariner: x[0], x[1] are velocity perturbations around U0.
        u_total = U_des + v[0]
        v_total = v[1]
        speed_ship = np.hypot(u_total, v_total)

        vwx = v_wind * np.cos(psi_wind)
        vwy = v_wind * np.sin(psi_wind)
        urx = vwx - speed_ship * np.cos(psi)
        ury = vwy - speed_ship * np.sin(psi)
        v_rel = np.hypot(urx, ury)
        gamma_r = np.arctan2(ury, urx) - psi

        tau_w, _, _, _ = isherwood72(
            gamma_r=gamma_r,
            V_r=v_rel,
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
        return tau_w

    return _wind_force


def make_wave_force_func(wave_conf=None):
    if wave_conf is None:
        return lambda _t, _v: np.zeros(3, dtype=float)

    wave_h = wave_conf.get("H", 0.0)
    wave_t = wave_conf.get("T", 8.0)
    wave_beta_deg = wave_conf.get("beta_deg", 0.0)
    if wave_h <= 0.0 or wave_t <= 0.0:
        return lambda _t, _v: np.zeros(3, dtype=float)

    w0 = 2.0 * np.pi / max(wave_t, 1e-6)
    nw = 21
    w = np.linspace(0.5 * w0, 2.0 * w0, nw)
    rng = np.random.default_rng(123)
    fai = rng.uniform(0.0, 2.0 * np.pi, size=nw)
    wave_beta = np.radians(wave_beta_deg)

    def _wave_force(t_sec, v):
        psi = v[5]
        speed_ship = np.hypot(U_des + v[0], v[1])
        beta_r = wave_beta - psi
        tau_wave = waveforce_irregular(
            t=t_sec,
            L=L_ship,
            h=wave_h,
            T=wave_t,
            beta_r=beta_r,
            w=w,
            fai=fai,
            U=speed_ship,
        )
        return np.asarray(tau_wave, dtype=float)

    return _wave_force


def make_current_drift_func(current_conf=None):
    if current_conf is None:
        return lambda _t, _v: (0.0, 0.0)

    v_c_mps = current_conf.get("Vc_mps", 0.0)
    beta_c_deg = current_conf.get("beta_c_deg", 0.0)
    beta_c = np.radians(beta_c_deg)

    def _current_drift(_t, v):
        psi = v[5]
        v_c_nd = v_c_mps / U_des
        u_c_nd, v_c_nd_body = decompose_current(beta_c=beta_c, V_c=v_c_nd, psi=psi, U0=U_des)
        u_c = u_c_nd * U_des
        v_c = v_c_nd_body * U_des
        xdot_c = np.cos(psi) * u_c - np.sin(psi) * v_c
        ydot_c = np.sin(psi) * u_c + np.cos(psi) * v_c
        return float(xdot_c), float(ydot_c)

    return _current_drift


def run_simulation(with_disturb=False, tf=1000.0, dt=0.1, rudder_deg=35.0,
                   wind_speed=12.0, wind_dir_deg=45.0,
                   wave_h=2.0, wave_t=8.0, wave_dir_deg=135.0,
                   current_speed=0.5, current_dir_deg=155.0):
    ship = ShipParams()
    state = np.zeros(7, dtype=float)
    delta_c = np.radians(rudder_deg)

    if with_disturb:
        wind_conf = dict(V_wind=wind_speed, Psi_wind_deg=wind_dir_deg)
        wave_conf = dict(H=wave_h, T=wave_t, beta_deg=wave_dir_deg)
        current_conf = dict(Vc_mps=current_speed, beta_c_deg=current_dir_deg)
        wind_force_func = make_wind_force_func(wind_conf=wind_conf, ship=ship)
        wave_force_func = make_wave_force_func(wave_conf=wave_conf)
        current_drift_func = make_current_drift_func(current_conf=current_conf)
    else:
        wind_force_func = None
        wave_force_func = None
        current_drift_func = None

    n = int((tf - 0.0) / dt) + 1
    traj = np.zeros((n, 8), dtype=float)  # [t, state(7)]
    t = 0.0

    def _mariner_ode(t_sec, x, cmd, wf, wavf, cdf):
        tau_total = np.zeros(3, dtype=float)
        if wf is not None:
            tau_total += wf(t_sec, x)
        if wavf is not None:
            tau_total += wavf(t_sec, x)
        tau_input = tau_total if (wf is not None or wavf is not None) else None

        xdot = mariner(x, cmd, U0=U_des, wind_force=tau_input)
        if cdf is not None:
            xdot_c, ydot_c = cdf(t_sec, x)
            xdot[3] += xdot_c
            xdot[4] += ydot_c
        return xdot

    for i in range(n):
        traj[i, 0] = t
        traj[i, 1:] = state
        state = rk4(_mariner_ode, t, state, dt, delta_c, wind_force_func, wave_force_func, current_drift_func)
        t += dt

    return traj


def main():
    parser = argparse.ArgumentParser(description="Mariner 回转测试（有/无干扰）")
    parser.add_argument("--units", choices=["m", "nd"], default="m", help="坐标单位：m 或 nd")
    parser.add_argument("--tf", type=float, default=300.0, help="仿真总时长（秒）")
    parser.add_argument("--dt", type=float, default=0.1, help="积分步长（秒）")
    parser.add_argument("--rudder-deg", type=float, default=35.0, help="恒定舵角（度）")
    parser.add_argument("--wind-speed", type=float, default=12.0, help="干扰风速（m/s）")
    parser.add_argument("--wind-dir", type=float, default=45.0, help="干扰风向（度）")
    parser.add_argument("--wave-h", type=float, default=2.0, help="干扰浪高 H（m）")
    parser.add_argument("--wave-t", type=float, default=8.0, help="干扰浪周期 T（s）")
    parser.add_argument("--wave-dir", type=float, default=135.0, help="干扰浪向（度）")
    parser.add_argument("--current-speed", type=float, default=0.5, help="干扰流速（m/s）")
    parser.add_argument("--current-dir", type=float, default=155.0, help="干扰流向（度）")
    args = parser.parse_args()

    traj_no = run_simulation(
        with_disturb=False,
        tf=args.tf,
        dt=args.dt,
        rudder_deg=args.rudder_deg,
        wind_speed=args.wind_speed,
        wind_dir_deg=args.wind_dir,
        wave_h=args.wave_h,
        wave_t=args.wave_t,
        wave_dir_deg=args.wave_dir,
        current_speed=args.current_speed,
        current_dir_deg=args.current_dir,
    )
    traj_yes = run_simulation(
        with_disturb=True,
        tf=args.tf,
        dt=args.dt,
        rudder_deg=args.rudder_deg,
        wind_speed=args.wind_speed,
        wind_dir_deg=args.wind_dir,
        wave_h=args.wave_h,
        wave_t=args.wave_t,
        wave_dir_deg=args.wave_dir,
        current_speed=args.current_speed,
        current_dir_deg=args.current_dir,
    )

    # mariner 输出的 x, y 为米制坐标
    x_no_m, y_no_m = traj_no[:, 4], traj_no[:, 5]
    x_yes_m, y_yes_m = traj_yes[:, 4], traj_yes[:, 5]

    if args.units == "nd":
        x_no, y_no = x_no_m / L_ship, y_no_m / L_ship
        x_yes, y_yes = x_yes_m / L_ship, y_yes_m / L_ship
        xlabel, ylabel = "横坐标 (无量纲)", "纵坐标 (无量纲)"
    else:
        x_no, y_no = x_no_m, y_no_m
        x_yes, y_yes = x_yes_m, y_yes_m
        xlabel, ylabel = "横坐标 (米)", "纵坐标 (米)"

    print("No-disturbance final position:", float(x_no[-1]), float(y_no[-1]))
    print("With-disturbance final position:", float(x_yes[-1]), float(y_yes[-1]))
    print("Final headings (rad):", float(traj_no[-1, 6]), float(traj_yes[-1, 6]))

    try:
        import matplotlib.pyplot as plt

        font_size = 12
        plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.labelsize"] = font_size
        plt.rcParams["xtick.labelsize"] = font_size
        plt.rcParams["ytick.labelsize"] = font_size
        plt.rcParams["legend.fontsize"] = font_size
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure()
        plt.plot(x_no, y_no, color="black", linestyle="-", label="无干扰")
        plt.plot(x_yes, y_yes, color="blue", linestyle="--", label="有干扰")
        plt.axis("equal")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    except Exception as e:
        print("Matplotlib not available or failed to plot:", e)


if __name__ == "__main__":
    main()
