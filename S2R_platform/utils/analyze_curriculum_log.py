import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class LogRow:
    episode: int
    progress: float
    ss4_ratio_target: float
    is_forced_ss4: int


def split_runs(rows: List[LogRow]) -> List[List[LogRow]]:
    if not rows:
        return []
    runs: List[List[LogRow]] = [[rows[0]]]
    for row in rows[1:]:
        prev = runs[-1][-1]
        if row.episode <= prev.episode:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


def read_curriculum_rows(path: Path) -> tuple[List[LogRow], int]:
    rows: List[LogRow] = []
    skipped_old_format = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            if raw[0].strip().lower() == "episode":
                continue
            # New format has the 2 appended columns:
            # ..., current_dir_deg, ss4_ratio_target, is_forced_ss4
            if len(raw) < 15:
                skipped_old_format += 1
                continue
            try:
                row = LogRow(
                    episode=int(float(raw[0])),
                    progress=float(raw[5]),
                    ss4_ratio_target=float(raw[13]),
                    is_forced_ss4=int(float(raw[14])),
                )
            except ValueError:
                continue
            rows.append(row)

    return rows, skipped_old_format


def summarize(rows: List[LogRow], tail_ratio: float) -> str:
    if not rows:
        return "没有检测到包含 `ss4_ratio_target/is_forced_ss4` 的新格式行。"

    runs = split_runs(rows)
    run = runs[-1]
    n = len(run)
    tail_n = max(1, int(n * tail_ratio))
    tail = run[-tail_n:]

    overall_forced_rate = sum(r.is_forced_ss4 for r in run) / n
    tail_forced_rate = sum(r.is_forced_ss4 for r in tail) / tail_n
    overall_target_mean = sum(r.ss4_ratio_target for r in run) / n
    tail_target_mean = sum(r.ss4_ratio_target for r in tail) / tail_n
    target_last = run[-1].ss4_ratio_target

    lines = [
        "=== Curriculum DR 生效性分析（最新一段训练）===",
        f"样本行数: {n}（后{int(tail_ratio * 100)}%窗口: {tail_n}）",
        f"进度范围: {run[0].progress:.4f} -> {run[-1].progress:.4f}",
        f"目标SS4占比(全程均值): {overall_target_mean:.4f}",
        f"目标SS4占比(后{int(tail_ratio * 100)}%均值): {tail_target_mean:.4f}",
        f"目标SS4占比(最后一行): {target_last:.4f}",
        f"实际强制SS4占比(全程): {overall_forced_rate:.4f}",
        f"实际强制SS4占比(后{int(tail_ratio * 100)}%): {tail_forced_rate:.4f}",
        f"后{int(tail_ratio * 100)}%差值(实际-目标): {tail_forced_rate - tail_target_mean:+.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze curriculum logging effect from training_log.csv"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/training_log.csv",
        help="Path to training CSV log",
    )
    parser.add_argument(
        "--tail-ratio",
        type=float,
        default=0.2,
        help="Tail window ratio used for effectiveness check (default: 0.2)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    rows, skipped_old = read_curriculum_rows(log_path)
    print(summarize(rows, tail_ratio=max(0.01, min(1.0, args.tail_ratio))))
    if skipped_old > 0:
        print(f"\n提示: 检测到 {skipped_old} 行旧格式日志（无新增两列），已自动跳过。")


if __name__ == "__main__":
    main()
