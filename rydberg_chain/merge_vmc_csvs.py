#!/usr/bin/env python3
"""
将多次训练的 parsed/summary CSV 合并为一份大 CSV，便于画整条训练曲线（含微调续训）。

用法:
  python3 merge_vmc_csvs.py [目录] [--name 基名] [--precision complex64|complex128]
  python3 merge_vmc_csvs.py --precision complex64
  python3 merge_vmc_csvs.py rydberg_chain/train/complex64
  python3 merge_vmc_csvs.py rydberg_chain/train/complex128 --name rydberg_L16_delta0.5_Rb1.0_alpha6

不指定目录时，默认用 train/<precision>（--precision 默认 complex128）。
会扫描目录下 {name}_run1_parsed.csv, {name}_run2_parsed.csv, ...（按 run 编号排序），
合并为：
  - {name}_merged_parsed.csv：列 run, global_iter, iter, Energy, sigma_E, Mx, Mz, Ntot, accept
    （global_iter 为跨 run 的连续迭代编号，便于把多次训练视为一条长轨迹）
  - {name}_merged_summary.csv：每行一个 run 的摘要，带 run 列
"""

import argparse
import csv
import os
import re
import sys

# --- 默认超参数（可由命令行覆盖）---
DEFAULT_PRECISION = "complex128"
DEFAULT_L = 16
DEFAULT_Rb = 1.0
DEFAULT_DELTA = 0.5
DEFAULT_ALPHA = 6.0


def _find_run_files(base_dir: str, name: str, suffix: str):
    """列出 {name}_run{N}{suffix}.csv，返回 [(N, path), ...] 按 N 排序。"""
    pattern = re.compile(rf"^{re.escape(name)}_run(\d+){re.escape(suffix)}\.csv$")
    out = []
    if not os.path.isdir(base_dir):
        return out
    for f in os.listdir(base_dir):
        m = pattern.match(f)
        if m:
            out.append((int(m.group(1)), os.path.join(base_dir, f)))
    return sorted(out, key=lambda x: x[0])


def merge_parsed(base_dir: str, name: str, out_path: str) -> int:
    """
    合并所有 {name}_run{N}_parsed.csv，写入 out_path。
    增加列 run, global_iter（跨 run 连续编号，便于整条训练曲线）。
    返回合并的 run 数量。
    """
    runs = _find_run_files(base_dir, name, "_parsed")
    if not runs:
        return 0

    rows_all = []
    global_offset = 0
    for run_num, path in runs:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows_all.append({
                    "run": run_num,
                    "global_iter": global_offset,
                    **row,
                })
                global_offset += 1

    out_headers = ["run", "global_iter"] + list(fieldnames)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_all)

    return len(runs)


def merge_summary(base_dir: str, name: str, out_path: str) -> int:
    """合并所有 {name}_run{N}_summary.csv，写入 out_path，增加 run 列。"""
    runs = _find_run_files(base_dir, name, "_summary")
    if not runs:
        return 0

    rows_all = []
    fieldnames = None
    for run_num, path in runs:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["run"] + reader.fieldnames
            for row in reader:
                row["run"] = run_num
                rows_all.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_all)

    return len(runs)


def main():
    parser = argparse.ArgumentParser(
        description="合并多次训练的 parsed/summary CSV 为一份大 CSV（含 run、global_iter）。"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="存放 *_run*_parsed.csv 的目录（不指定时用 train/<precision>）",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        choices=("complex64", "complex128"),
        default=DEFAULT_PRECISION,
        help=f"精度子目录名（默认: {DEFAULT_PRECISION}）",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L, help=f"链长（默认: {DEFAULT_L}）")
    parser.add_argument("--Rb", type=float, default=DEFAULT_Rb, help=f"Blockade radius（默认: {DEFAULT_Rb}）")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help=f"Detuning（默认: {DEFAULT_DELTA}）")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"相互作用指数 α（默认: {DEFAULT_ALPHA}）")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="文件名基名；不指定时由 L/Rb/delta/alpha 构造",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_subdir = f"L{args.L}_Rb{args.Rb}_delta{args.delta}_alpha{args.alpha}"
    name = args.name if args.name is not None else f"rydberg_L{args.L}_delta{args.delta}_Rb{args.Rb}_alpha{args.alpha}"
    if args.directory:
        base_dir = os.path.abspath(args.directory)
    else:
        base_dir = os.path.join(script_dir, "train", args.precision, param_subdir)

    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在 {base_dir}", file=sys.stderr)
        sys.exit(1)

    out_parsed = os.path.join(base_dir, f"{name}_merged_parsed.csv")
    out_summary = os.path.join(base_dir, f"{name}_merged_summary.csv")

    n_parsed = merge_parsed(base_dir, name, out_parsed)
    n_summary = merge_summary(base_dir, name, out_summary)

    if n_parsed == 0:
        print(f"未找到 {name}_run*_parsed.csv，请检查目录与 --name。", file=sys.stderr)
        sys.exit(1)

    print(f"已合并 {n_parsed} 次训练的 parsed → {out_parsed}")
    print(f"  （列含 run, global_iter；global_iter 为跨 run 的连续迭代编号）")
    if n_summary > 0:
        print(f"已合并 {n_summary} 次训练的 summary → {out_summary}")
    else:
        print(f"未找到 {name}_run*_summary.csv，未生成合并摘要。")
    print("完成。")


if __name__ == "__main__":
    main()
