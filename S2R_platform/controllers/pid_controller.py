import math
from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float = 5.0
    ki: float = 0.0
    kd: float = 0.0


class PIDController:
    """
    PID controller with optional anti-windup via integral clamp.
    """

    def __init__(
        self,
        gains: PIDGains,
        i_limit: float | None = None,
        use_rate_as_derivative: bool = True,
    ) -> None:
        self.gains = gains
        self.i_limit = i_limit
        self.use_rate_as_derivative = use_rate_as_derivative
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def compute(self, error: float, yaw_rate: float, dt: float) -> float:
        """Return control in radians for given heading error and yaw rate."""
        self._integral += error * dt
        if self.i_limit is not None:
            self._integral = float(max(-self.i_limit, min(self.i_limit, self._integral)))

        if self.use_rate_as_derivative:
            d_term = -yaw_rate
        else:
            if self._prev_error is None or dt <= 0:
                d_term = 0.0
            else:
                d_term = (error - self._prev_error) / dt
        self._prev_error = error

        return (
            -self.gains.kp * error
            - self.gains.ki * self._integral
            - self.gains.kd * d_term
        )


def pd_control(error: float, yaw_rate: float, kp: float, td: float) -> float:
    """PD control that matches the existing formulation: -Kp*(e + Td*r)."""
    return -kp * (error + td * yaw_rate)
