import numpy as np


def waveforce_irregular(t, L, h, T, beta_r, w, fai, U):
    rho = 1025
    g = 9.8
    c = 48.93549412
    w = w + (w ** 2 / g) * 1.1 * np.sqrt(c) * np.cos(beta_r)
    delta_w = (np.max(w) - np.min(w)) / len(w)

    A = 8.1 * np.exp(-3)
    B = 3.11 / (h ** 2)

    fwx, fwy, fwn = 0.0, 0.0, 0.0

    for i in range(len(w)):
        wi = w[i]
        fai_i = fai[i]
        Sw = A / wi ** 5 * np.exp(-B / wi ** 4)
        lamda = 2 * np.pi / wi ** 2 * g

        Cxw = 0.05 - 0.2 * (lamda / L) + 0.75 * (lamda / L) ** 2 - 0.51 * (lamda / L) ** 3
        Cyw = 0.46 + 6.83 * (lamda / L) - 15.65 * (lamda / L) ** 2 + 8.44 * (lamda / L) ** 3
        Cnw = -0.11 + 0.68 * (lamda / L) - 0.79 * (lamda / L) ** 2 + 0.21 * (lamda / L) ** 3

        fwx += 0.5 * rho * g * L * np.cos(beta_r) * abs(Cxw) * (2 * Sw * delta_w) \
               + 1.0 * rho * g * L * np.cos(beta_r) * Cxw * (2 * Sw * delta_w) * np.cos(wi * t + fai_i)
        fwy += 0.5 * rho * g * L * np.sin(beta_r) * abs(Cyw) * (2 * Sw * delta_w) \
               + 1.0 * rho * g * L * np.sin(beta_r) * Cyw * (2 * Sw * delta_w) * np.cos(wi * t + fai_i)
        fwn += 0.5 * rho * g * L ** 2 * np.sin(beta_r) * abs(Cnw) * (2 * Sw * delta_w) \
               + 1.0 * rho * g * L ** 2 * np.sin(beta_r) * Cnw * (2 * Sw * delta_w) * np.cos(wi * t + fai_i)

    # Return SI units (N, N, N·m); nondimensionalization is handled by caller
    tau_wave = np.array([fwx, fwy, fwn])
    return tau_wave


def wave_model(method, **kwargs):
    """
    简化的波浪响应模型，兼容旧接口。

    返回:
        z_heave (m), phi_roll_deg (deg), theta_pitch_deg (deg), wave_state
    """
    wave_state = kwargs.get('wave_state', None)
    if method is None:
        return 0.0, 0.0, 0.0, wave_state

    t = float(kwargs.get('t', 0.0))
    wave_a = float(kwargs.get('wave_a', 0.0))  # 振幅 (m)
    wave_T0 = float(kwargs.get('wave_T0', kwargs.get('T_0', 8.0)))
    phase = float(kwargs.get('phase', 0.0))

    if wave_T0 <= 1e-6 or wave_a == 0.0:
        return 0.0, 0.0, 0.0, wave_state

    omega = 2 * np.pi / wave_T0
    z_heave = wave_a * np.sin(omega * t + phase)
    # 简单滚转响应：与波浪斜率成比例，幅值适当缩放
    phi_roll_rad = 0.1 * wave_a * omega * np.cos(omega * t + phase)
    phi_roll_deg = np.degrees(phi_roll_rad)
    theta_pitch_deg = 0.0
    return z_heave, phi_roll_deg, theta_pitch_deg, wave_state
