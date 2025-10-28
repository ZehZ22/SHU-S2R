import numpy as np

def generate_straight_path(length, angle_deg, interval):
    """
    Generate straight path from origin to length at given angle.
    Ensures the last waypoint is exactly at `length` even when length < interval.
    """
    angle_rad = np.radians(angle_deg)
    n = max(2, int(np.ceil(max(length, 1e-9) / max(interval, 1e-9))) + 1)
    x = np.linspace(0.0, length, n)
    y = np.tan(angle_rad) * x
    return list(zip(x, y))

def generate_random_line_path(r_min, r_max, interval):
    """
    Generate a straight path from origin to a random endpoint uniformly
    sampled in an annulus with radius in [r_min, r_max] and angle in [0, 2π).
    All quantities are nondimensional (L=1).
    """
    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(0.0, 2 * np.pi)
    return generate_straight_path(length=r, angle_deg=np.degrees(theta), interval=interval)

def generate_rectangle_path(length, width, interval):
    # 生成矩形路径：起点 -> 右 -> 上 -> 左 -> 下回到起点
    path = []

    # 底边
    for x in np.arange(0, length, interval):
        path.append((x, 0))

    # 右边
    for y in np.arange(0, width, interval):
        path.append((length, y))

    # 顶边
    for x in np.arange(length, 0, -interval):
        path.append((x, width))

    # 左边
    for y in np.arange(width, 0, -interval):
        path.append((0, y))

    return path

def generate_circular_path(radius, interval_angle_deg):
    angles = np.radians(np.arange(0, 360, interval_angle_deg))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return list(zip(x, y))

def generate_s_curve_path():
    """
    Fixed S-shaped waypoint sequence as in the spec:
    Adjusted first waypoint to ensure each segment length > 2.5L when R_switch=2.5L.
    start at (0,0) then (0,2.6), (3,4), (5,2), (7,4) in nondimensional units (L=1).
    """
    return [(0.0, 0.0), (1.0, 2.0), (3.0, 4.0), (5.0, 2.0), (7.0, 4.0)]

class PathManager:
    def __init__(self, waypoints):
        self.path = waypoints
        self.start_x = waypoints[0][0]
        self.start_y = waypoints[0][1]
        self.start_psi = np.arctan2(waypoints[1][1] - waypoints[0][1],
                                    waypoints[1][0] - waypoints[0][0])
        self.current_index = 1  # 初始目标点索引

    def calculate_heading_error(self, pos):
        x, y, psi = pos
        wp0 = self.path[self.current_index - 1]
        wp1 = self.path[self.current_index]

        path_angle = np.arctan2(wp1[1] - wp0[1], wp1[0] - wp0[0])
        heading_error = path_angle - psi
        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        return heading_error

    def calculate_cross_track_error(self, pos):
        x, y, _ = pos
        wp0 = self.path[self.current_index - 1]
        wp1 = self.path[self.current_index]

        dx = wp1[0] - wp0[0]
        dy = wp1[1] - wp0[1]
        path_angle = np.arctan2(dy, dx)

        distance = np.hypot(x - wp0[0], y - wp0[1])
        angle_to_point = np.arctan2(y - wp0[1], x - wp0[0])
        cross_track_error = np.sin(path_angle - angle_to_point) * distance
        return cross_track_error

    def is_finished(self, x, y, tolerance=50):
        dx = self.path[-1][0] - x
        dy = self.path[-1][1] - y
        return np.hypot(dx, dy) < tolerance

    def update_waypoint(self, x, y, threshold=50):
        # 当到达当前目标点后，更新下一个目标点
        if self.current_index >= len(self.path):
            return  # 已到终点

        target_wp = self.path[self.current_index]
        distance_to_target = np.hypot(target_wp[0] - x, target_wp[1] - y)

        if distance_to_target < threshold and self.current_index < len(self.path) - 1:
            self.current_index += 1
