import numpy as np

def current(x, y, U0):
    """
    动态计算洋流速度和方向。
    """
    V_c = 1.5 / U0  # 动态速度，例如随位置变化
    V_angle = np.radians(155)  # 动态方向角
    return V_c, V_angle


def decompose_current(beta_c, V_c, psi, U0):
    """
    Resolve ambient current into ship-body components (nondimensional).

    Parameters
    - beta_c: current direction in earth frame (radians, from x axis)
    - V_c: current speed nondimensionalized by U0 (i.e., Vc_mps / U0)
    - psi: ship heading (radians)
    - U0: reference speed in m/s (typically U_des)

    Returns
    - (u_c, v_c): body-frame nondimensional components to be used for
      both kinematics (added to position rate) and hydrodynamics (subtracted
      from ship velocities as relative flow where applicable).
    """
    x = np.cos(beta_c) * V_c * U0
    y = np.sin(beta_c) * V_c * U0
    u_c = (np.cos(psi) * x - np.sin(psi) * y) / U0
    v_c = (np.sin(psi) * x + np.cos(psi) * y) / U0
    return u_c, v_c

def decompose_current_mps(beta_c, Vc_mps, psi, U0):
    """Helper that accepts current speed in m/s and returns nondimensional
    body components referenced to U0.
    """
    V_nd = Vc_mps / U0
    return decompose_current(beta_c, V_nd, psi, U0)
