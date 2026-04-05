import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CURVES = {
    'NDR': os.path.join('logs', 'ndr_training.csv'),
    'LDR': os.path.join('logs', 'ldr_training.csv'),
    'HDR': os.path.join('logs', 'hdr_training.csv'),
    'CRDR': os.path.join('logs', 'crdr_training.csv'),
}


def load_curve(csv_path: str) -> dict:
    episodes = []
    steps = []
    avg_return = []
    return_std = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required = {'episode', 'total_steps', 'ep_return', 'avg_return', 'return_std'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns in {csv_path}: {sorted(missing)}')
        for row in reader:
            episodes.append(float(row['episode']))
            steps.append(float(row['total_steps']))
            avg_return.append(float(row['avg_return']))
            return_std.append(float(row['return_std']))
    return {
        'episode': episodes,
        'total_steps': steps,
        'avg_return': avg_return,
        'return_std': return_std,
    }


def smooth_curve_by_bins(series: dict, metric: str, *, bin_size: float) -> tuple[list[float], list[float]]:
    x = np.asarray(series['total_steps'], dtype=float)
    y = np.asarray(series[metric], dtype=float)
    if x.size == 0:
        return [], []
    bin_ids = np.floor((x - x.min()) / max(bin_size, 1.0)).astype(int)
    x_out = []
    y_out = []
    for bid in np.unique(bin_ids):
        mask = bin_ids == bid
        x_out.append(float(x[mask].mean()))
        y_out.append(float(y[mask].mean()))
    return x_out, y_out


def smooth_curve_by_moving_average(series: dict, metric: str, *, window_size: int) -> tuple[list[float], list[float]]:
    x = np.asarray(series['total_steps'], dtype=float)
    y = np.asarray(series[metric], dtype=float)
    if x.size == 0:
        return [], []
    w = max(1, int(window_size))
    if x.size < w:
        return x.tolist(), y.tolist()
    kernel = np.ones(w, dtype=float) / float(w)
    x_out = np.convolve(x, kernel, mode='valid')
    y_out = np.convolve(y, kernel, mode='valid')
    return x_out.tolist(), y_out.tolist()


def plot_metric(curves: dict, metric: str, ylabel: str, title: str, *, x_key: str, x_label: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, series in curves.items():
        plt.plot(series[x_key], series[metric], linewidth=2, label=label)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_smoothed(
    curves: dict,
    metric: str,
    ylabel: str,
    title: str,
    *,
    smooth_mode: str,
    bin_size: float,
    moving_window: int,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, series in curves.items():
        if smooth_mode == 'moving':
            x_plot, y_plot = smooth_curve_by_moving_average(
                series, metric, window_size=moving_window
            )
        else:
            x_plot, y_plot = smooth_curve_by_bins(
                series, metric, bin_size=bin_size
            )
        plt.plot(x_plot, y_plot, linewidth=2, label=label)
    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(
    x_axis_mode: str = 'steps',
    smooth: bool = False,
    smooth_mode: str = 'bin',
    bin_size: float = 10000.0,
    moving_window: int = 10,
):
    parser = argparse.ArgumentParser(description='Compare training curves from multiple strategies.')
    parser.add_argument('--ndr', type=str, default=DEFAULT_CURVES['NDR'])
    parser.add_argument('--ldr', type=str, default=DEFAULT_CURVES['LDR'])
    parser.add_argument('--hdr', type=str, default=DEFAULT_CURVES['HDR'])
    parser.add_argument('--crdr', type=str, default=DEFAULT_CURVES['CRDR'])
    args = parser.parse_args()

    curve_paths = {
        'NDR': args.ndr,
        'LDR': args.ldr,
        'HDR': args.hdr,
        'CRDR': args.crdr,
    }

    curves = {}
    for label, csv_path in curve_paths.items():
        if not os.path.exists(csv_path):
            print(f'[WARN] Missing CSV for {label}: {csv_path}')
            continue
        curves[label] = load_curve(csv_path)

    if not curves:
        print('No valid CSV files were found.')
        return

    if x_axis_mode == 'episode':
        x_key = 'episode'
        x_label = 'Episode'
        avg_title = 'Average Return vs Episode'
        std_title = 'Return STD vs Episode'
    else:
        x_key = 'total_steps'
        x_label = 'Training Steps'
        if smooth:
            if smooth_mode == 'moving':
                avg_title = f'Average Return vs Training Steps (Moving Average, window={moving_window})'
                std_title = f'Return STD vs Training Steps (Moving Average, window={moving_window})'
            else:
                avg_title = f'Average Return vs Training Steps (Bin Average, bin={int(bin_size)})'
                std_title = f'Return STD vs Training Steps (Bin Average, bin={int(bin_size)})'
        else:
            avg_title = 'Average Return vs Training Steps'
            std_title = 'Return STD vs Training Steps'

    if x_axis_mode == 'steps' and smooth:
        plot_metric_smoothed(
            curves,
            'avg_return',
            'Average Return',
            avg_title,
            smooth_mode=smooth_mode,
            bin_size=bin_size,
            moving_window=moving_window,
        )
        plot_metric_smoothed(
            curves,
            'return_std',
            'Return STD',
            std_title,
            smooth_mode=smooth_mode,
            bin_size=bin_size,
            moving_window=moving_window,
        )
        return

    plot_metric(curves, 'avg_return', 'Average Return', avg_title, x_key=x_key, x_label=x_label)
    plot_metric(curves, 'return_std', 'Return STD', std_title, x_key=x_key, x_label=x_label)


if __name__ == '__main__':
    # Change to 'episode' if you want the x-axis to use training episodes instead of steps.
    X_AXIS_MODE = 'steps'
    # Smoothing is only applied when X_AXIS_MODE == 'steps'.
    ENABLE_SMOOTH = True
    # Options: 'bin' or 'moving'
    SMOOTH_MODE = 'bin'
    # Used when SMOOTH_MODE == 'bin'
    BIN_SIZE = 10000.0
    # Used when SMOOTH_MODE == 'moving'
    MOVING_WINDOW = 10
    main(
        x_axis_mode=X_AXIS_MODE,
        smooth=ENABLE_SMOOTH,
        smooth_mode=SMOOTH_MODE,
        bin_size=BIN_SIZE,
        moving_window=MOVING_WINDOW,
    )
