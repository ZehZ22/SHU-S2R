import argparse
import csv
import io
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vessels.kcs import L as L_ship

plt.rcParams["font.family"] = ["Times New Roman", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220
plt.rcParams["savefig.dpi"] = 300

MAIN_FONT_SIZE = 20
TICK_FONT_SIZE = 16

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

POLICY_ORDER = ["NDR", "S2S3DR", "S3S4DR", "DR", "CRDR"]
POLICY_CN = {
    "NDR": "无随机化",
    "S2S3DR": "海况2到海况3域随机化",
    "S3S4DR": "海况3到海况4域随机化",
    "DR": "海况2到海况4随机化",
    "CRDR": "课程学习式随机化",
}
LINE_STYLES = {
    "NDR": dict(color="tab:green", linestyle="-.", marker="o"),
    "DR": dict(color="tab:blue", linestyle="--", marker="s"),
    "CRDR": dict(color="tab:red", linestyle="-", marker="^"),
    "S2S3DR": dict(color="tab:purple", linestyle=":", marker="D"),
    "S3S4DR": dict(color="tab:orange", linestyle=(0, (5, 2)), marker="P"),
}


def load_rows(csv_path: str | None) -> List[Dict[str, str]]:
    if csv_path:
        with open(csv_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_CSV
    return list(csv.DictReader(io.StringIO(text.strip())))


def convert_y1_to_m(rows: List[Dict[str, str]], length_m: float) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        nr = dict(r)
        nr["y1_MAE"] = f"{float(r['y1_MAE']) * length_m:.6f}"
        out.append(nr)
    return out


def plot_y1_mae_ss1_to_ss4(rows: List[Dict[str, str]]):
    # Remove SS5 and keep SS1~SS4 only
    sea_states = ["SS1", "SS2", "SS3", "SS4"]
    sea_label_map = {"SS1": "海况1", "SS2": "海况2", "SS3": "海况3", "SS4": "海况4"}
    rows = [r for r in rows if r["SeaState"] in sea_states]

    x = np.arange(len(sea_states))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

    for policy in POLICY_ORDER:
        y_vals = []
        for sea in sea_states:
            found = next((r for r in rows if r["SeaState"] == sea and r["Policy"] == policy), None)
            y_vals.append(float(found["y1_MAE"]) if found else np.nan)
        if not np.all(np.isnan(y_vals)):
            ax.plot(
                x,
                y_vals,
                label=POLICY_CN.get(policy, policy),
                linewidth=2.2,
                markersize=6.8,
                **LINE_STYLES.get(policy, dict(linestyle="-", marker="o")),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([sea_label_map[s] for s in sea_states])
    ax.set_xlabel(" ", fontsize=MAIN_FONT_SIZE)
    ax.set_ylabel("侧偏距平均绝对误差 (m)", fontsize=MAIN_FONT_SIZE)
    ax.set_ylim(0, 55)
    ax.tick_params(axis="both", labelsize=MAIN_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=MAIN_FONT_SIZE)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="侧偏距平均绝对误差可视化（SS1~SS4，m）")
    parser.add_argument("--csv", type=str, default=None, help="可选：外部CSV路径")
    parser.add_argument("--length-m", type=float, default=float(L_ship), help="船长L（米）")
    args = parser.parse_args()

    rows_nd = load_rows(args.csv)
    rows_m = convert_y1_to_m(rows_nd, length_m=args.length_m)
    plot_y1_mae_ss1_to_ss4(rows_m)


if __name__ == "__main__":
    main()
