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
