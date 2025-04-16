import numpy as np
from .wavesigma_func import wavesigma_func
from .wlambda_func import Wlambda_Func

# === wave_func1：基于waveresponse ===
def wave_func1(t, a, beta, T_0, zeta4, T4, GMT, Cb, U, L, B, T, **kwargs):
    g = 9.81  # 重力加速度
    rho_water = kwargs.get('rho_water', 1025)

    nabla = Cb * L * B * T
    w_0 = 2 * np.pi / T_0
    k = w_0 ** 2 / g
    w_e = w_0 - k * U * np.cos(beta)
    k_e = abs(k * np.cos(beta))
    sigma = k_e * L / 2
    kappa = np.exp(-k_e * T)

    alpha = w_e / w_0
    A = 2 * np.sin(k * B * alpha ** 2 / 2) * np.exp(-k * T * alpha ** 2)
    f = np.sqrt((1 - k * T) ** 2 + (A ** 2 / (k * B * alpha ** 3)) ** 2)
    F = kappa * f * np.sin(sigma) / sigma
    G = kappa * f * (6 / L) * (1 / sigma) * (np.sin(sigma) / sigma - np.cos(sigma))

    wn = np.sqrt(g / (2 * T))
    zeta = (A ** 2 / (B * alpha ** 3)) * np.sqrt(1 / (8 * k ** 3 * T))

    Z3 = np.sqrt((2 * wn * zeta) ** 2 + (1 / w_e ** 2) * (wn ** 2 - w_e ** 2) ** 2)
    eps3 = np.arctan2(2 * w_e * wn * zeta, wn ** 2 - w_e ** 2)
    z_heave = (a * F * wn ** 2 / (Z3 * w_e)) * np.cos(w_e * t + eps3)

    Z5 = Z3
    eps5 = eps3
    theta_pitch = (a * G * wn ** 2 / (Z5 * w_e)) * np.sin(w_e * t + eps5)
    theta_pitch_deg = np.degrees(theta_pitch)

    w4 = 2 * np.pi / T4
    C44 = rho_water * g * nabla * GMT
    M44 = C44 / w4 ** 2
    B44 = 2 * zeta4 * w4 * M44
    M = np.sin(beta) * np.sqrt(B44 * rho_water * g ** 2 / max(w_e, 1e-6))

    Z4 = np.sqrt((2 * w4 * zeta4) ** 2 + (1 / w_e ** 2) * (w4 ** 2 - w_e ** 2) ** 2)
    eps4 = np.arctan2(2 * w_e * w4 * zeta4, w4 ** 2 - w_e ** 2)
    phi_roll = ((M / C44) * w4 ** 2 / (Z4 * w_e)) * np.cos(w_e * t + eps4)
    phi_roll_deg = np.degrees(phi_roll)

    return z_heave, phi_roll_deg, theta_pitch_deg, None  # 保持接口一致，多返回一个 None 占位


# === wave_func2：高频波浪响应，基于Xie的Wave_Func 中的高频波浪力

def fwave_func(Wlambda, omega_e, sigma, z_state, eta_wave_state, gau):
    dz_dt = -sigma * z_state + omega_e ** 2 * eta_wave_state + gau
    return dz_dt

def wave_func2(t, dt, wave_state, wind_speed, z_0, eta, nu, Psi_wind,
               gau_noise, ship,
               wave_a, wave_beta, wave_T0, wave_zeta4, wave_T4):
    """
    wave_func2 动态波浪响应模型
    """

    # === 实时更新 Wlambda 和 sigma_wave ===
    Wlambda, sigma_wave = Wlambda_Func(
        V_wind=wind_speed,
        z_0=z_0,
        eta=eta,
        nu=nu,
        Psi_wind=Psi_wind
    )

    # === 状态解包 ===
    Heta_wave, z_state = wave_state

    # 角速度向量
    omega_wave = float(np.atleast_1d(Wlambda).flatten()[0])

    # 激励力（简化处理为正弦变化）
    Heta_wave_new = Heta_wave + dt * z_state
    z_state_new = z_state - (2 * wave_zeta4 * omega_wave * z_state + omega_wave ** 2 * Heta_wave) * dt

    # 波浪升沉量（垂向位移）
    z_heave = Heta_wave_new * np.cos(omega_wave * t)

    # 横摇角（度）
    phi_roll = Heta_wave_new * np.sin(omega_wave * t) * 5.0
    phi_roll_deg = np.degrees(phi_roll)

    # 纵倾角（度）
    theta_pitch = Heta_wave_new * np.sin(omega_wave * t + np.pi / 4) * 3.0
    theta_pitch_deg = np.degrees(theta_pitch)

    # 更新状态
    updated_wave_state = (Heta_wave_new, z_state_new)

    return z_heave, phi_roll_deg, theta_pitch_deg, updated_wave_state




# === 统一接口 ===
def wave_model(method, *args, **kwargs):
    if method == 'func1':
        return wave_func1(*args, **kwargs)
    elif method == 'func2':
        return wave_func2(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported wave model method: {method}")
