import argparse
import numpy as np

from vessels.kcs import (
    KCS_ode,
    L, d_em, rho, U_des,
    Dp, n_prop, wp, tp, a0, a1, a2,
)
from disturbances.wind import isherwood72
from disturbances.wave import waveforce_irregular
from disturbances.current import decompose_current


def rk4(f, t, y, h, *args, **kwargs):
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args, **kwargs)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args, **kwargs)
    k4 = f(t + h, y + h * k3, *args, **kwargs)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_ext_force(
    wind_on=True, wind_speed=1.0, wind_dir_deg=45.0,
    wave_on=True, wave_H=1.0, wave_T=8.0, wave_dir_deg=45.0,
    cancel_prop=True,
):
    """Build an ext_force callback with optional wind/wave components.

    Returns a function ext_force(t_nd, v, _) -> (Xp, Yp, Np) in nondimensional form.
    """
    scale_F = 0.5 * rho * L * d_em * (U_des ** 2)
    scale_N = 0.5 * rho * (L ** 2) * d_em * (U_des ** 2)

    rng = np.random.default_rng(123)
    Nw = 21
    def _ext_force(t_nd, v, _):
        psi = v[5]
        up = v[0]

        X_SI = 0.0
        Y_SI = 0.0
        N_SI = 0.0

        # 1) Wind (optional)
        if wind_on and wind_speed > 0:
            Psi_wind = np.radians(wind_dir_deg)
            gamma_r = Psi_wind - psi
            tauW, *_ = isherwood72(
                gamma_r=gamma_r,
                V_r=wind_speed,
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
            X_SI += tauW[0]
            Y_SI += tauW[1]
            N_SI += tauW[2]

        # 2) Wave (optional)
        if wave_on and wave_H > 0 and wave_T > 0:
            w0 = 2 * np.pi / wave_T
            w = np.linspace(0.8 * w0, 1.2 * w0, Nw)
            fai = rng.uniform(0, 2 * np.pi, size=Nw)
            beta_r = np.radians(wave_dir_deg) - psi
            t_sec = t_nd * (L / U_des)
            tau_wave = waveforce_irregular(
                t=t_sec, L=L, h=wave_H, T=wave_T, beta_r=beta_r, w=w, fai=fai, U=up * U_des,
            )
            X_SI += tau_wave[0]
            Y_SI += tau_wave[1]
            N_SI += tau_wave[2]

        # 3) Propulsion cancellation (optional)
        Xp_cancel = 0.0
        if cancel_prop:
            J = (max(up, 0.0) * U_des) * (1 - wp) / (max(n_prop * Dp, 1e-9))
            Kt = a0 + a1 * J + a2 * (J ** 2)
            X_P = (1 - tp) * rho * Kt * (Dp ** 4) * (n_prop ** 2)
            Xp_cancel = X_P / scale_F

        # Convert to nondimensional
        Xp = X_SI / scale_F - Xp_cancel
        Yp = Y_SI / scale_F
        Np = N_SI / scale_N
        return np.array([Xp, Yp, Np])

    return _ext_force


def make_current_func(current_on=True, V_env=1.0, dir_deg=45.0):
    def _cur(t_nd, v, _):
        if not current_on or V_env <= 0:
            return 0.0, 0.0
        psi = v[5]
        beta_c = np.radians(dir_deg)
        V_c_nd = V_env / U_des
        u_c, v_c = decompose_current(beta_c=beta_c, V_c=V_c_nd, psi=psi, U0=U_des)
        return u_c, v_c
    return _cur


def run(duration_s=600.0, dt_nd=0.1, units='nd',
        wind_on=True, wind_speed=1.0, wind_dir=45.0,
        wave_on=True, wave_H=1.0, wave_T=8.0, wave_dir=45.0,
        current_on=True, current_speed=1.0, current_dir=45.0,
        cancel_prop=True):
    v = np.zeros(7)
    delta_c = 0.0

    ext_force = make_ext_force(
        wind_on=wind_on, wind_speed=wind_speed, wind_dir_deg=wind_dir,
        wave_on=wave_on, wave_H=wave_H, wave_T=wave_T, wave_dir_deg=wave_dir,
        cancel_prop=cancel_prop,
    )
    current_func = make_current_func(current_on=current_on, V_env=current_speed, dir_deg=current_dir)

    nd_total_time = duration_s / (L / U_des)
    steps = int(np.ceil(nd_total_time / dt_nd))

    # Preallocate one extra row to include initial state at t=0
    t = 0.0
    traj = np.zeros((steps + 1, 8))
    cur_hist = np.zeros((steps + 1, 2))  # u_c, v_c per step

    # Save initial state
    traj[0, 0] = t
    traj[0, 1:] = v
    cur_hist[0, :] = current_func(t, v, None)

    for i in range(1, steps + 1):
        # Log current contribution before stepping (for diagnostics)
        u_c, v_c = current_func(t, v, None)
        cur_hist[i, 0] = u_c
        cur_hist[i, 1] = v_c

        v = rk4(KCS_ode, t, v, dt_nd,
                delta_c,
                ext_force=ext_force,
                ext_ctx=None,
                current_func=current_func,
                current_ctx=None)
        t += dt_nd
        traj[i, 0] = t
        traj[i, 1:] = v

    # State ordering stored in traj as: [t, up, vp, rp, x, y, psi, delta]
    x_nd, y_nd = traj[:, 4], traj[:, 5]
    if units == 'm':
        x, y = x_nd * L, y_nd * L
        xlabel, ylabel = 'x (m)', 'y (m)'
    else:
        x, y = x_nd, y_nd
        xlabel, ylabel = 'x (nd)', 'y (nd)'

    try:
        import matplotlib.pyplot as plt
        # Trajectory plot
        plt.figure(figsize=(7, 5))
        plt.plot(x, y, label='Drift under wind/wave/current')
        plt.plot(x[:1], y[:1], 'ko', label='Start')
        plt.axis('equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title('KCS Drift Trajectory (delta=0, initial rest)')

        # Time histories of u, v, r, delta, psi
        t_arr = traj[:, 0]
        up_arr = traj[:, 1]
        vp_arr = traj[:, 2]
        rp_arr = traj[:, 3]
        x_arr = traj[:, 4]
        y_arr = traj[:, 5]
        psi_arr = np.degrees(traj[:, 6])
        delta_arr = np.degrees(traj[:, 7])

        fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
        ax[0].plot(t_arr, up_arr); ax[0].set_ylabel('u (nd)')
        ax[1].plot(t_arr, vp_arr); ax[1].set_ylabel('v (nd)')
        ax[2].plot(t_arr, rp_arr); ax[2].set_ylabel('r (nd)')
        ax[3].plot(t_arr, delta_arr); ax[3].set_ylabel('delta (deg)')
        ax[4].plot(t_arr, psi_arr); ax[4].set_ylabel('psi (deg)'); ax[4].set_xlabel('t (nd)')
        fig.suptitle('State Time Histories')

        plt.show()
    except Exception as e:
        print('Plot failed:', e)

    # Print diagnostic snapshot
    print('Diagnostics (first 5 steps):')
    for i in range(min(6, traj.shape[0])):
        t_i = traj[i, 0]
        up_i, vp_i, rp_i = traj[i, 1], traj[i, 2], traj[i, 3]
        x_i, y_i, psi_i, delta_i = traj[i, 4], traj[i, 5], traj[i, 6], traj[i, 7]
        uc_i, vc_i = cur_hist[i, 0], cur_hist[i, 1]
        print(f"t_nd={t_i:.3f}, up={up_i:.3f}, vp={vp_i:.3f}, rp={rp_i:.3f}, x={x_i:.3f}, y={y_i:.3f}, psi={psi_i:.3f}, delta={delta_i:.3f}, u_c={uc_i:.3f}, v_c={vc_i:.3f}")

    # No CSV export requested


def main():
    p = argparse.ArgumentParser(description='Environment drift experiment for KCS')
    p.add_argument('--time-s', type=float, default=600.0, help='Physical duration in seconds')
    p.add_argument('--dt', type=float, default=0.1, help='Nondimensional time step (L/U_des)')
    p.add_argument('--units', choices=['nd', 'm'], default='nd', help='Plot units')
    # Wind
    p.add_argument('--wind', action='store_true', default=True, help='Enable wind')
    p.add_argument('--wind-speed', type=float, default=4.0, help='Wind speed (m/s)')
    p.add_argument('--wind-dir', type=float, default=60.0, help='Wind direction (deg)')
    # Wave
    p.add_argument('--wave', action='store_true', default=True, help='Enable wave')
    p.add_argument('--wave-H', type=float, default=4.0, help='Significant wave height H (m)')
    p.add_argument('--wave-T', type=float, default=8.0, help='Wave period T (s)')
    p.add_argument('--wave-dir', type=float, default=70.0, help='Wave direction (deg)')
    # Current
    p.add_argument('--current', action='store_true', default=True, help='Enable current')
    p.add_argument('--current-speed', type=float, default=2.5, help='Current speed (m/s)')
    p.add_argument('--current-dir', type=float, default=100.0, help='Current direction (deg)')
    # Propulsion cancellation
    p.add_argument('--cancel-prop', action='store_true', default=True, help='Cancel propulsion force (engine-off)')
    args = p.parse_args()

    run(duration_s=args.time_s, dt_nd=args.dt, units=args.units,
        wind_on=args.wind, wind_speed=args.wind_speed, wind_dir=args.wind_dir,
        wave_on=args.wave, wave_H=args.wave_H, wave_T=args.wave_T, wave_dir=args.wave_dir,
        current_on=args.current, current_speed=args.current_speed, current_dir=args.current_dir,
        cancel_prop=args.cancel_prop)


if __name__ == '__main__':
    main()
