import numpy as np
from scipy.optimize import curve_fit
from .wavesigma_func import wavesigma_func



def S_omega_func(omega, lambda_param, H_s, T_1, gamma, wave_sigma):
    """
    S_omega 经验公式部分，对应 MATLAB Wlambda_Func.m 中 S_omega 计算
    """
    exponent = -((0.191 * omega * T_1 - 1) / (np.sqrt(2) * wave_sigma)) ** 2
    Y = np.exp(exponent)
    return 155 * H_s ** 2 / T_1 ** 4 * omega ** -5 * np.exp(-944 / T_1 ** 4 * omega ** -4) * gamma ** Y


def Pyy_Func(omega, lambda_param, omega_e, sigma):
    """
    Pyy 函数，对应 MATLAB Pyy_Func.m 第一部分公式
    """
    numerator = 4 * (lambda_param * omega_e * sigma) ** 2 * omega ** 2
    denominator = (omega_e ** 2 - omega ** 2) ** 2 + 4 * (lambda_param * omega_e * omega) ** 2
    return numerator / denominator


def Wlambda_Func(V_wind, z_0, eta, nu, Psi_wind):
    """
    高频海浪谱参数计算，来源于 Wlambda_Func.m
    输入：
    - V_wind: 风速 (m/s)
    - z_0: 风速计安装高度 (m)
    - eta: 位姿向量 [x, y, yaw]
    - nu: 速度向量 [u, v, r]
    - Psi_wind: 风向 (rad)

    输出：
    - Wlambda: 拟合得到的 lambda
    - sigma_wave: 波能谱 sigma
    """
    g = 9.81  # 重力加速度
    z = 19.4  # 风速计高度

    # 风速剖面模型（幂律）
    V_wind_z = V_wind * (z / z_0) ** (1 / 7)

    # 有义波高估算公式（经验公式）
    H_s = 2.06 / g ** 2 * V_wind_z ** 2

    # 峰值周期，固定值
    T_0 = 7
    T_1 = 0.834 * T_0

    # 船舶遇波频率（ω_e），参考方向风速
    U = nu[0]
    beta = Psi_wind - eta[2]
    omega_0 = 2 * np.pi / T_0
    omega_e = abs(omega_0 - (omega_0 ** 2 / g) * U * np.cos(beta))

    # 频率范围
    N = 100
    omega = np.linspace(0.01, 3, N)

    # gamma 系数
    gamma = 3.3

    # 计算波能谱 sigma_wave（根据经验公式）
    S_omega = np.zeros(N)

    for i in range(N):
        wave_sigma = wavesigma_func(omega[i], T_1)
        S_omega[i] = S_omega_func(omega[i], 1.0, H_s, T_1, gamma, wave_sigma)

    # sigma_wave 取最大谱密度的平方根
    sigma_wave = np.sqrt(np.max(S_omega))

    # 使用 Pyy 函数进行 lambda 拟合
    popt, _ = curve_fit(
        lambda omega, lambda_param: Pyy_Func(omega, lambda_param, omega_e, sigma_wave),
        omega, S_omega, p0=[0.1]
    )
    Wlambda = popt[0]

    return Wlambda, sigma_wave
