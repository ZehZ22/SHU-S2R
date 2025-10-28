import numpy as np
import argparse

from vessels.kcs import KCS_ode, L, d_em, rho, U_des
from disturbances.wind import isherwood72
from disturbances.wave import waveforce_irregular
from disturbances.current import decompose_current
from ship_params import ShipParams


def rk4(f, t, y, h, *args, **kwargs):
    k1 = f(t, y, *args, **kwargs)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args, **kwargs)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args, **kwargs)
    k4 = f(t + h, y + h * k3, *args, **kwargs)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def disturbance_func(wind_conf=None, wave_conf=None, ship=None):
    scale_F = 0.5 * rho * L * d_em * (U_des ** 2)
    scale_N = 0.5 * rho * (L ** 2) * d_em * (U_des ** 2)

    def _ext_force(t_nd, v, _):
        # Convert nondimensional time to seconds for disturbance models if needed
        t_sec = t_nd * (L / U_des)
        psi = v[5]
        up = v[0]
        U_ship = up * U_des

        X_SI = 0.0
        Y_SI = 0.0
        N_SI = 0.0

        if wind_conf is not None and ship is not None:
            V_wind = wind_conf.get('V_wind', 0.0)
            Psi_wind = np.radians(wind_conf.get('Psi_wind_deg', 0.0))
            gamma_r = Psi_wind - psi
            tauW, CX, CY, CN = isherwood72(
                gamma_r=gamma_r,
                V_r=V_wind,
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
            X_SI += tauW[0]
            Y_SI += tauW[1]
            N_SI += tauW[2]

        if wave_conf is not None:
            h = wave_conf.get('H', 0.0)
            T = wave_conf.get('T', 10.0)
            beta_wave = np.radians(wave_conf.get('beta_deg', 0.0))
            w0 = 2 * np.pi / max(T, 1e-6)
            # Build a narrow-banded wave spectrum around w0
            Nw = 21
            w = np.linspace(0.8 * w0, 1.2 * w0, Nw)
            rng = np.random.default_rng(123)
            fai = rng.uniform(0, 2 * np.pi, size=Nw)
            beta_r = beta_wave - psi
            tau_wave = waveforce_irregular(
                t=t_sec, L=L, h=h, T=T, beta_r=beta_r, w=w, fai=fai, U=U_ship,
            )
            X_SI += tau_wave[0]
            Y_SI += tau_wave[1]
            N_SI += tau_wave[2]

        # Convert to nondimensional Abkowitz form expected by KCS_ode
        Xp = X_SI / scale_F
        Yp = Y_SI / scale_F
        Np = N_SI / scale_N
        return np.array([Xp, Yp, Np])

    return _ext_force


def make_current_func(Vc_mps=0.0, beta_c_deg=0.0):
    def _cur(t_nd, v, _):
        psi = v[5]
        beta_c = np.radians(beta_c_deg)
        V_c_nd = (Vc_mps / U_des)  # nondimensional ratio with U_des
        u_c, v_c = decompose_current(beta_c=beta_c, V_c=V_c_nd, psi=psi, U0=U_des)
        return u_c, v_c
    return _cur


def run_simulation(with_disturb=False):
    ship = ShipParams()

    # Initial state: [up, vp, rp, x, y, psi, delta]
    v = np.zeros(7)
    v[0] = 1.0  # up
    delta_c = np.radians(35.0)

    # Time in nondimensional units
    t0, tf, dt = 0.0, 300.0, 0.1
    N = int((tf - t0) / dt) + 1
    t = t0

    traj = np.zeros((N, 8))  # t, state(7)

    if with_disturb:
        wind_conf = dict(V_wind=12.0, Psi_wind_deg=45.0)
        wave_conf = dict(H=2.0, T=8.0, beta_deg=135.0, phase=0.0)
        ext_force = disturbance_func(wind_conf=wind_conf, wave_conf=wave_conf, ship=ship)
        current_func = make_current_func(Vc_mps=0.5, beta_c_deg=155.0)
    else:
        ext_force = None
        current_func = None

    for i in range(N):
        traj[i, 0] = t
        traj[i, 1:] = v
        v = rk4(KCS_ode, t, v, dt,
                delta_c,
                ext_force=ext_force,
                ext_ctx=None,
                current_func=current_func,
                current_ctx=None)
        t += dt

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', choices=['nd', 'm'], default='nd', help='Plot units: nd (default) or meters')
    args = parser.parse_args()

    traj_no = run_simulation(with_disturb=False)
    traj_yes = run_simulation(with_disturb=True)

    # Non-dimensional state positions
    x_no_nd, y_no_nd = traj_no[:, 4], traj_no[:, 5]
    x_yes_nd, y_yes_nd = traj_yes[:, 4], traj_yes[:, 5]

    if args.units == 'm':
        from vessels.kcs import L as L_ship
        x_no, y_no = x_no_nd * L_ship, y_no_nd * L_ship
        x_yes, y_yes = x_yes_nd * L_ship, y_yes_nd * L_ship
        xlabel, ylabel = 'x (m)', 'y (m)'
    else:
        x_no, y_no = x_no_nd, y_no_nd
        x_yes, y_yes = x_yes_nd, y_yes_nd
        xlabel, ylabel = 'x (nd)', 'y (nd)'

    print("No-disturbance final position:", float(x_no[-1]), float(y_no[-1]))
    print("With-disturbance final position:", float(x_yes[-1]), float(y_yes[-1]))
    print("Final headings (rad):", float(traj_no[-1, 6]), float(traj_yes[-1, 6]))

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_no, y_no, label='No Disturbance')
        plt.plot(x_yes, y_yes, label='Wind/Wave/Current')
        plt.axis('equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title('KCS Turning Circle, delta=35 deg')
        plt.show()
    except Exception as e:
        print("Matplotlib not available or failed to plot:", e)


if __name__ == '__main__':
    main()

