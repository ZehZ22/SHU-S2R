import numpy as np


def mariner(x, ui, U0=7.7175, wind_force=None):
    """
    Mariner model for USV with wind effects using isherwood72.

    :param x: state vector [u, v, r, x, y, psi, delta]
    :param ui: commanded rudder angle (rad)
    :param U0: nominal speed, default is 7.7175 m/s (15 knots)
    :return: time derivative of the state vector and speed U
    """

    if len(x) != 7:
        raise ValueError('x-vector must have dimension 7 !')
    if not isinstance(ui, (int, float)):
        raise ValueError('ui must be a scalar input!')

    # Normalization variables
    L = 160.93
    U = np.sqrt((U0 + x[0]) ** 2 + x[1] ** 2)

    # Non-dimensional states and inputs
    delta_c = -ui  # delta_c = -ui such that positive delta_c -> positive r

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
    Xuu = -110e-5
    Yv = -1160e-5;
    Nv = -264e-5
    Xuuu = -215e-5;
    Yr = -499e-5;
    Nr = -166e-5
    Xvv = -899e-5;
    Yvvv = -8078e-5;
    Nvvv = 1636e-5
    Xrr = 18e-5;
    Yvvr = 15356e-5;
    Nvvr = -5483e-5
    Xdd = -95e-5;
    Yvu = -1160e-5;
    Nvu = -264e-5
    Xudd = -190e-5;
    Yru = -499e-5;
    Nru = -166e-5
    Xrv = 798e-5;
    Yd = 278e-5;
    Nd = -139e-5
    Xvd = 93e-5;
    Yddd = -90e-5;
    Nddd = 45e-5
    Xuvd = 93e-5;
    Yud = 556e-5;
    Nud = -278e-5
    Yuud = 278e-5;
    Nuud = -139e-5
    Yvdd = -4e-5;
    Nvdd = 13e-5
    Yvvd = 1190e-5;
    Nvvd = -489e-5
    Y0 = -4e-5;
    N0 = 3e-5
    Y0u = -8e-5;
    N0u = 6e-5
    Y0uu = -4e-5;
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

    # Forces and moments
    X = Xu * u + Xuu * u ** 2 + Xuuu * u ** 3 + Xvv * v ** 2 + Xrr * r ** 2 + Xrv * r * v + Xdd * delta ** 2 + \
        Xudd * u * delta ** 2 + Xvd * v * delta + Xuvd * u * v * delta

    Y = Yv * v + Yr * r + Yvvv * v ** 3 + Yvvr * v ** 2 * r + Yvu * v * u + Yru * r * u + Yd * delta + \
        Yddd * delta ** 3 + Yud * u * delta + Yuud * u ** 2 * delta + Yvdd * v * delta ** 2 + \
        Yvvd * v ** 2 * delta + (Y0 + Y0u * u + Y0uu * u ** 2)

    N = Nv * v + Nr * r + Nvvv * v ** 3 + Nvvr * v ** 2 * r + Nvu * v * u + Nru * r * u + Nd * delta + \
        Nddd * delta ** 3 + Nud * u * delta + Nuud * u ** 2 * delta + Nvdd * v * delta ** 2 + \
        Nvvd * v ** 2 * delta + (N0 + N0u * u + N0uu * u ** 2)

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

    xdot = np.zeros(7)
    xdot[0] = X * (U ** 2 / L) / m11
    xdot[1] = -(-m33 * Y + m23 * N) * (U ** 2 / L) / detM22
    xdot[2] = (-m32 * Y + m22 * N) * (U ** 2 / L ** 2) / detM22
    xdot[3] = (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U
    xdot[4] = (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U
    xdot[5] = r * (U / L)
    xdot[6] = delta_dot

    return np.array(xdot)
