#!/usr/bin/env python3
"""
将多次训练的 parsed/summary CSV 合并为一份大 CSV，便于画整条训练曲线。
（与 rydberg_chain 相同逻辑，默认目录与基名为 long_range_ising/train/<precision>/ 下的 rbm_LongIsing_*）

用法:
  python3 merge_vmc_csvs.py [目录] [--name 基名] [--precision complex64|complex128]
  python3 merge_vmc_csvs.py --precision complex64
  python3 merge_vmc_csvs.py --delta 0   # 合并 delta=0 的 CSV（默认）
  python3 merge_vmc_csvs.py --delta 0.5 --precision complex128
  python3 merge_vmc_csvs.py long_range_ising/train/complex64 --name rbm_LongIsing_L=16_J=1.0_delta=0.0_...

不指定目录时，默认用 train/<precision>（--precision 默认 complex64）。
会扫描目录下 {name}_run1_parsed.csv, {name}_run2_parsed.csv, ...，
合并为 {name}_merged_parsed.csv 与 {name}_merged_summary.csv。
"""

import argparse
import csv
import os
import re
import sys

# 默认精度子目录，与 parse_vmc_log / rbm_long_range_ising 一致
DIGIT = "complex64"


def _basename_from_delta(delta: float, L: int = 16) -> str:
    """与 rbm_long_range_ising 输出文件名一致。"""
    return f"rbm_LongIsing_L={L}_J=1.0_delta={delta}_alphaInt=2.0_alpha=4_Cal1"


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
    """合并所有 {name}_run{N}_parsed.csv，写入 out_path。增加列 run, global_iter。"""
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
        description="合并多次训练的 parsed/summary CSV 为一份大 CSV（Long-range Ising，含 run、global_iter）。"
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
        default=DIGIT,
        help=f"精度子目录名，用于 train/<precision>（默认: {DIGIT}）",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="纵场 detuning，用于构造默认基名，与 rbm_long_range_ising 一致（默认: 0.0）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="文件名基名，不含 _runN；不指定时由 --delta 构造",
    )
    args = parser.parse_args()

    name = args.name if args.name is not None else _basename_from_delta(args.delta)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.directory:
        base_dir = os.path.abspath(args.directory)
    else:
        base_dir = os.path.join(script_dir, "train", args.precision)

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
    print(f"  （列含 run, global_iter）")
    if n_summary > 0:
        print(f"已合并 {n_summary} 次训练的 summary → {out_summary}")
    else:
        print(f"未找到 {name}_run*_summary.csv，未生成合并摘要。")
    print("完成。")


if __name__ == "__main__":
    main()
