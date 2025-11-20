import numpy as np


def ILOSpsi(x, y, Delta, kappa, h, U, R_switch, wpt, psi=None):
    """
    Computes the desired yaw angle and yaw rate for straight-line paths
    through the waypoints.
    """
    # Initialization of persistent variables
    if not hasattr(ILOSpsi, "k"):
        ILOSpsi.k = None
        ILOSpsi.y_int = 0
    if not hasattr(ILOSpsi, "verbose"):
        ILOSpsi.verbose = False

    if ILOSpsi.k is None:
        # Check if R_switch is smaller than the minimum distance between the waypoints
        dists = np.sqrt(np.diff(wpt['x']) ** 2 + np.diff(wpt['y']) ** 2)
        if R_switch > np.min(dists):
            raise ValueError("The distances between the waypoints must be larger than R_switch")
        if R_switch < 0:
            raise ValueError("R_switch must be larger than zero")
        if Delta < 0:
            raise ValueError("Delta must be larger than zero")

        ILOSpsi.y_int = 0  # Integral state
        ILOSpsi.k = 0  # Set first waypoint as the active waypoint

    # Read next waypoint
    n = len(wpt['x'])
    if ILOSpsi.k < n - 1:
        xk_next = wpt['x'][ILOSpsi.k + 1]
        yk_next = wpt['y'][ILOSpsi.k + 1]
    else:
        xk_next = wpt['x'][-1]
        yk_next = wpt['y'][-1]

    # Print active waypoint (optional)
    xk = wpt['x'][ILOSpsi.k]
    yk = wpt['y'][ILOSpsi.k]
    if ILOSpsi.verbose:
        print(f"Active waypoint:\n  (x{ILOSpsi.k + 1}, y{ILOSpsi.k + 1}) = ({xk:.2f}, {yk:.2f})")

    # Compute the desired yaw angle
    pi_p = np.arctan2(yk_next - yk, xk_next - xk)  # Path-tangential angle w.r.t. North

    # Along-track and cross-track errors
    x_e = (x - xk) * np.cos(pi_p) + (y - yk) * np.sin(pi_p)
    y_e = -(x - xk) * np.sin(pi_p) + (y - yk) * np.cos(pi_p)

    # If the next waypoint satisfies the switching criterion, k = k + 1
    d = np.sqrt((xk_next - xk) ** 2 + (yk_next - yk) ** 2)
    if (d - x_e < R_switch) and (ILOSpsi.k < n - 1):
        ILOSpsi.k += 1
        xk = xk_next  # Update active waypoint
        yk = yk_next

    # ILOS guidance law
    Kp = 1 / Delta
    Ki = kappa * Kp
    psi_d = pi_p - np.arctan(Kp * y_e + Ki * ILOSpsi.y_int)

    # Kinematic differential equation
    Dy_int = Delta * y_e / (Delta ** 2 + (y_e + kappa * ILOSpsi.y_int) ** 2)

    # Euler integration
    ILOSpsi.y_int += h * Dy_int

    if psi is not None:
        omega_psi_d = -(Kp * Dy_int + Ki * ILOSpsi.y_int) / ((Kp * y_e + Ki * ILOSpsi.y_int) ** 2 + 1)
        return psi_d, omega_psi_d
    else:
        return psi_d
