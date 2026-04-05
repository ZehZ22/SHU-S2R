"""
侧偏距平均绝对误差对比（分组输出）：
- 图1：NDR vs CRDR
- 图2：含无域随机化在内的全部策略对比（顺序同 POLICY_ORDER）
"""
import argparse
import csv
import io
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vessels.kcs import L as L_ship

# 中文宋体（SimSun）；纵轴刻度数字用 Times New Roman；字号与图尺寸统一
FIG_SIZE = (7.2, 5.2)
BASE_FONT_PT = 11
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimSun", "NSimSun", "Times New Roman", "DejaVu Sans"],
        "font.size": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT,
        "ytick.labelsize": BASE_FONT_PT,
        "legend.fontsize": BASE_FONT_PT,
        "axes.unicode_minus": False,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "figure.figsize": FIG_SIZE,
    }
)

DEFAULT_CSV = """SeaState,Policy,y1_MAE,y1_STD,y1_MAX_ABS,y2_MAE(rad),y2_STD(rad),y2_MAX_ABS(rad),delta_MAE(deg),delta_STD(deg),delta_MAX_ABS(deg)
SS1,CRDR,0.0952,0.1628,1.0593,0.1115,0.1211,0.4784,13.0652,16.2872,34.1919
SS1,DR,0.1082,0.1869,1.0432,0.1132,0.1533,0.6101,14.6923,17.7806,34.6186
SS1,NDR,0.1271,0.1890,0.9712,0.0702,0.1160,0.4126,5.0982,9.5714,28.7421
SS2,CRDR,0.0933,0.1665,1.1207,0.1094,0.1197,0.5054,13.4251,16.4308,34.3788
SS2,DR,0.1005,0.1771,1.0941,0.1052,0.1389,0.6218,14.5515,17.5187,34.6297
SS2,NDR,0.4042,0.4606,1.3780,0.2030,0.2333,0.9205,6.7398,10.0374,29.7417
SS3,CRDR,0.1373,0.2050,1.1246,0.1207,0.1399,0.6861,15.8425,18.2114,34.8108
SS3,DR,0.1530,0.2260,1.1324,0.1300,0.1656,0.7196,17.3108,19.7533,34.6960
SS3,NDR,0.5071,0.5593,1.3425,0.2363,0.2584,0.8227,7.6223,10.2758,29.8795
SS4,CRDR,0.1928,0.2549,1.1826,0.1247,0.1406,0.5569,16.5039,18.4885,34.5691
SS4,DR,0.2180,0.2875,1.3109,0.1485,0.1811,0.9072,18.8868,21.0910,34.7547
SS4,NDR,0.6344,0.6704,1.6177,0.2668,0.2813,0.9759,8.2605,10.5624,30.0512
SS5,CRDR,0.3024,0.3779,1.5992,0.1499,0.1706,0.7667,17.1540,19.3498,34.8520
SS5,DR,0.3300,0.4069,1.9676,0.1717,0.2009,0.8974,19.8306,22.0152,34.7320
SS5,NDR,0.8170,0.8531,2.4398,0.2717,0.2966,1.0796,9.2320,11.5473,31.2652
SS1,S2S3DR,0.0930,0.1692,1.0375,0.1057,0.1419,0.5450,14.4835,18.1223,34.4633
SS2,S2S3DR,0.0884,0.1645,1.0728,0.0969,0.1223,0.5364,14.5333,17.8519,34.4775
SS3,S2S3DR,0.1415,0.2114,1.1173,0.1117,0.1435,0.6661,16.3044,19.2291,34.5067
SS4,S2S3DR,0.2100,0.2817,1.1417,0.1172,0.1530,0.7042,15.8312,18.8097,34.5360
SS5,S2S3DR,0.3724,0.4758,1.8978,0.1523,0.1940,0.8397,16.8086,19.8416,34.5658
SS1,S3S4DR,0.1155,0.1884,1.1056,0.1290,0.1471,0.5245,16.7541,19.3598,34.6252
SS2,S3S4DR,0.1012,0.1691,1.1040,0.1185,0.1293,0.6202,15.1815,18.1487,34.6418
SS3,S3S4DR,0.1384,0.2112,1.1212,0.1280,0.1510,0.6235,15.6066,18.9070,34.6457
SS4,S3S4DR,0.1984,0.2721,1.2080,0.1394,0.1647,0.6896,17.0216,19.9174,34.6339
SS5,S3S4DR,0.3115,0.3895,1.3412,0.1546,0.1765,0.7266,18.6648,21.4123,34.6399
"""

# 图注顺序：无域随机化 → 海况2–3 → 海况3–4 → 海况2–4 → 课程学习式域随机化
POLICY_ORDER = ["NDR", "S2S3DR", "S3S4DR", "DR", "CRDR"]
POLICY_CN = {
    "NDR": "无域随机化",
    "S2S3DR": "海况2到海况3域随机化",
    "S3S4DR": "海况3到海况4域随机化",
    "DR": "海况2到海况4域随机化",
    "CRDR": "课程学习式域随机化",
}

LINE_STYLES = {
    "NDR": dict(color="tab:green", linestyle="-.", marker="o"),
    "DR": dict(color="tab:blue", linestyle="--", marker="s"),
    "CRDR": dict(color="tab:red", linestyle="-", marker="^"),
    "S2S3DR": dict(color="tab:gray", linestyle=":", marker="D"),
    "S3S4DR": dict(color="tab:orange", linestyle=(0, (14, 6)), marker="P"),
}


def _sea_state_sort_key(sea_state: str) -> int:
    digits = "".join(ch for ch in sea_state if ch.isdigit())
    return int(digits) if digits else 0


def sea_states_to_xticklabels(sea_states: List[str]) -> List[str]:
    """SS1–SS5 → 海况1–海况5，其余保持原样。"""
    labels: List[str] = []
    for s in sea_states:
        if s.upper().startswith("SS"):
            suf = s[2:]
            if suf.isdigit():
                labels.append(f"海况{int(suf)}")
                continue
        labels.append(s)
    return labels


def load_rows(csv_path: str | None) -> List[Dict[str, str]]:
    if csv_path:
        with open(csv_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_CSV
    reader = csv.DictReader(io.StringIO(text.strip()))
    return list(reader)


def convert_y1_metrics_to_m(rows: List[Dict[str, str]], length_m: float) -> List[Dict[str, str]]:
    converted = []
    for row in rows:
        new_row = dict(row)
        for k in ("y1_MAE", "y1_STD", "y1_MAX_ABS"):
            new_row[k] = f"{float(row[k]) * length_m:.6f}"
        converted.append(new_row)
    return converted


def get_axis_values(
    rows: List[Dict[str, str]], policy_order: List[str], strict: bool = True
) -> tuple[List[str], List[str]]:
    sea_states = sorted({r["SeaState"] for r in rows}, key=_sea_state_sort_key)
    present_policies = {r["Policy"] for r in rows}
    policies = [p for p in policy_order if p in present_policies]
    if not strict:
        for p in sorted(present_policies):
            if p not in policies:
                policies.append(p)
    return sea_states, policies


def build_metric_matrix(
    rows: List[Dict[str, str]], sea_states: List[str], policies: List[str], metric_key: str
) -> np.ndarray:
    matrix = np.full((len(sea_states), len(policies)), np.nan, dtype=float)
    s_idx = {s: i for i, s in enumerate(sea_states)}
    p_idx = {p: i for i, p in enumerate(policies)}
    for r in rows:
        i = s_idx[r["SeaState"]]
        j = p_idx.get(r["Policy"])
        if j is not None:
            matrix[i, j] = float(r[metric_key])
    return matrix


def plot_y1_mae_lines(rows, sea_states, policies, y_label="侧偏距平均绝对误差 (m)"):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=220)
    x = np.arange(len(sea_states))
    mat = build_metric_matrix(rows, sea_states, policies, "y1_MAE")
    for k, policy in enumerate(policies):
        style = LINE_STYLES.get(policy, dict(color=None, linestyle="-", marker="o"))
        ax.plot(
            x,
            mat[:, k],
            label=POLICY_CN.get(policy, policy),
            linewidth=2.2,
            markersize=6.8,
            **style,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sea_states_to_xticklabels(sea_states))
    ax.set_xlabel("海况")
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    # 纵轴为数值：Times New Roman；横轴「海况1」等保持宋体
    for ylbl in ax.get_yticklabels():
        ylbl.set_fontfamily("Times New Roman")
    for xlbl in ax.get_xticklabels():
        xlbl.set_fontfamily("SimSun")
    ax.grid(axis="y", alpha=0.25)
    leg = ax.legend(loc="best")
    for t in leg.get_texts():
        t.set_fontfamily("SimSun")
    ax.xaxis.label.set_fontfamily("SimSun")
    ax.yaxis.label.set_fontfamily("SimSun")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="侧偏距平均绝对误差对比（分组）")
    parser.add_argument("--csv", type=str, default=None, help="可选：外部CSV路径")
    parser.add_argument(
        "--length-m",
        type=float,
        default=float(L_ship),
        help="船长L（米）",
    )
    args = parser.parse_args()

    rows_nd = load_rows(args.csv)
    rows_m = convert_y1_metrics_to_m(rows_nd, length_m=args.length_m)

    # 图1：NDR 与 CRDR（顺序与 POLICY_ORDER 中二者出现顺序一致）
    policies_ndr_crdr = [p for p in POLICY_ORDER if p in ("NDR", "CRDR")]
    sea_states, policies_1 = get_axis_values(rows_m, policies_ndr_crdr, strict=True)
    plot_y1_mae_lines(rows_m, sea_states, policies_1)

    # 图2：含无域随机化（NDR），图注顺序同 POLICY_ORDER
    policies_dr_group = list(POLICY_ORDER)
    sea_states, policies_2 = get_axis_values(rows_m, policies_dr_group, strict=True)
    plot_y1_mae_lines(rows_m, sea_states, policies_2)


if __name__ == "__main__":
    main()
