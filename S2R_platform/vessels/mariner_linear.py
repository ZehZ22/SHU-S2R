import numpy as np


def marinerlinear(x, ui, U0=7.7175, wind_force=None):
    if len(x) != 7:
        raise ValueError('x-vector must have dimension 7!')
    if not np.isscalar(ui):
        raise ValueError('ui must be a scalar input!')

    # Normalization variables
    L = 160.93
    U = np.sqrt((U0 + x[0]) ** 2 + x[1] ** 2)

    # Non-dimensional states and inputs
    delta_c = -ui

    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U
    psi = x[5]
    delta = x[6]

    # Parameters, hydrodynamic derivatives and main dimensions
    delta_max = 35  # max rudder angle (deg)
    Ddelta_max = 5  # max rudder derivative (deg/s)

    m = 798e-5
    Iz = 39.2e-5
    xG = -0.023

    Xudot = -42e-5
    Yvdot = -748e-5
    Nvdot = 4.646e-5
    Xu = -184e-5
    Yrdot = -9.354e-5
    Nrdot = -43.8e-5
    Yv = -1160e-5
    Nv = -264e-5
    Nr = -166e-5
    Yd = 278e-5
    Nd = -139e-5
    Y0 = -4e-5
    N0 = 3e-5
    Y0u = -8e-5
    N0u = 6e-5
    Y0uu = -4e-5
    N0uu = 3e-5

    # Masses and moments of inertia
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Iz - Nrdot

    # Rudder saturation and dynamics
    if abs(delta_c) >= delta_max * np.pi / 180:
        delta_c = np.sign(delta_c) * delta_max * np.pi / 180
    delta_dot = delta_c - delta
    if abs(delta_dot) >= Ddelta_max * np.pi / 180:
        delta_dot = np.sign(delta_dot) * Ddelta_max * np.pi / 180

    # Forces and moments (linearized)
    X = Xu * u
    Y = Yv * v + Yd * delta + (Y0 + Y0u * u)
    N = Nv * v + Nr * r + Nd * delta + (N0 + N0u * u)

    if wind_force is not None:
        # 无量纲化处理
        rho = 1000  # 水密度 (kg/m³)
        Loa = 160.93  # 船长 (m)
        q = 0.5 * rho * U ** 2 * Loa ** 2  # 动压面积项

        tau_w = wind_force.copy()
        tau_w[0] /= q
        tau_w[1] /= q
        tau_w[2] /= q * Loa  # 力矩需要乘 Loa

        X += tau_w[0]
        Y += tau_w[1]
        N += tau_w[2]
    # Dimensional state derivative
    detM22 = m22 * m33 - m23 * m32

    xdot = np.array([
        X * (U ** 2 / L) / m11,
        -(-m33 * Y + m23 * N) * (U ** 2 / L) / detM22,
        (-m32 * Y + m22 * N) * (U ** 2 / L ** 2) / detM22,
        (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U,
        (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U,
        r * (U / L),
        delta_dot
    ])

    return np.array(xdot)
