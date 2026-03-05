#!/usr/bin/env python3
"""
解析 NetKet VMC 生成的 .log 文件，输出便于阅读的摘要与表格，并可将结果保存为 CSV。

用法:
  python3 parse_vmc_log.py [log 文件路径]
  python3 parse_vmc_log.py   # 默认解析 train/complex128/rydberg_L16_delta0.5_Rb1.0_alpha6.log
  python3 parse_vmc_log.py --precision complex64   # 默认解析 train/complex64/ 下同名 log

输出:
  - 运行概览（迭代数、最终能量与误差、接受率等）
  - 观测量摘要（Mx, Mz, Ntot 等）
  - 可选：指定步数的能量与观测量表格
  - 默认保存为 name_runN_parsed.csv / name_runN_summary.csv（N 为第几次训练），不覆盖已有；-r N 可指定 N
  - -o/--csv：指定 CSV 路径时仍可覆盖
"""

import argparse
import csv
import json
import os
import re
import sys


def load_log(path: str) -> dict:
    """加载 JSON 格式的 log 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_energy_mean(entry: dict, i: int) -> float:
    """从 Energy 条目中取第 i 步的平均值（实部）。"""
    mean = entry.get("Mean")
    if isinstance(mean, dict) and "real" in mean:
        return mean["real"][i]
    if isinstance(mean, (list, tuple)):
        return mean[i] if i < len(mean) else float("nan")
    return float("nan")


def get_energy_sigma(entry: dict, i: int) -> float:
    """从 Energy 条目中取第 i 步的误差 Sigma。"""
    sigma = entry.get("Sigma", [])
    return sigma[i] if i < len(sigma) else float("nan")


def get_obs_mean(entry: dict, i: int) -> float:
    """从观测量条目（Mx/Mz/Ntot）中取第 i 步的 Mean（实部或标量）。"""
    mean = entry.get("Mean")
    if isinstance(mean, dict) and "real" in mean:
        arr = mean["real"]
        return arr[i] if i < len(arr) else float("nan")
    if isinstance(mean, (list, tuple)):
        return mean[i] if i < len(mean) else float("nan")
    return float("nan")


def get_acceptance(entry: dict, i: int) -> float:
    """从 acceptance 条目中取第 i 步的值。"""
    val = entry.get("value", entry.get("Mean", []))
    if isinstance(val, (list, tuple)):
        return val[i] if i < len(val) else float("nan")
    return float("nan")


def format_number(x: float, decimals: int = 6) -> str:
    if x != x:  # nan
        return "—"
    return f"{x: .{decimals}g}"


def print_summary(data: dict, log_path: str) -> None:
    """打印人类可读的摘要。"""
    n = len(data["Energy"]["iters"])
    last = n - 1

    print("=" * 60)
    print("  VMC 运行日志解析结果")
    print("=" * 60)
    print(f"  文件: {log_path}")
    print(f"  总迭代步数: {n}")
    print()

    # 能量
    E = data["Energy"]
    e_final = get_energy_mean(E, last)
    sig_final = get_energy_sigma(E, last)
    e_init = get_energy_mean(E, 0)
    print("  【能量 E = ⟨ψ|H|ψ⟩】")
    print(f"    初始 (iter=0):   {format_number(e_init)}")
    print(f"    最终 (iter={last}): {format_number(e_final)}  ±  {format_number(sig_final)}")
    print()

    # 接受率
    acc = data.get("acceptance", {})
    if acc:
        v = acc.get("value", acc.get("Mean", []))
        if isinstance(v, (list, tuple)) and len(v) > 0:
            a_final = v[last] if last < len(v) else v[-1]
            a_init = v[0]
            print("  【MC 接受率】")
            print(f"    初始: {format_number(a_init, 4)}")
            print(f"    最终: {format_number(a_final, 4)}")
            print()

    # 观测量
    obs_names = ["Mx", "Mz", "Ntot"]
    obs_desc = {"Mx": "横磁化 (1/L)Σ⟨σˣ⟩", "Mz": "纵磁化 (1/L)Σ⟨σᶻ⟩", "Ntot": "占据数 (1/L)Σ⟨n⟩"}
    print("  【观测量（最终步）】")
    for name in obs_names:
        if name not in data:
            continue
        val = get_obs_mean(data[name], last)
        desc = obs_desc.get(name, name)
        print(f"    {name} ({desc}): {format_number(val)}")
    print("=" * 60)


def get_table_rows(data: dict, steps: list) -> list[list]:
    """返回指定步数的数据行（用于 CSV 或打印）。每行为 [iter, Energy, sigma_E, Mx, Mz, Ntot, accept]。"""
    E = data["Energy"]
    acc = data.get("acceptance", {})
    rows = []
    for i in steps:
        if i >= len(E["iters"]):
            continue
        e_val = get_energy_mean(E, i)
        e_sig = get_energy_sigma(E, i)
        mx = get_obs_mean(data["Mx"], i) if "Mx" in data else float("nan")
        mz = get_obs_mean(data["Mz"], i) if "Mz" in data else float("nan")
        ntot = get_obs_mean(data["Ntot"], i) if "Ntot" in data else float("nan")
        a = get_acceptance(acc, i) if acc else float("nan")
        rows.append([i, e_val, e_sig, mx, mz, ntot, a])
    return rows


def print_table(data: dict, steps: list) -> None:
    """打印指定步数的能量与观测量表格。"""
    rows = get_table_rows(data, steps)
    headers = ["iter", "Energy", "σ(E)", "Mx", "Mz", "Ntot", "accept"]
    col_widths = [6, 14, 10, 12, 12, 12, 8]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        i, e_val, e_sig, mx, mz, ntot, a = r
        row_str = [
            str(i),
            format_number(e_val, 5),
            format_number(e_sig, 4),
            format_number(mx, 5),
            format_number(mz, 5),
            format_number(ntot, 5),
            format_number(a, 3),
        ]
        line = "  ".join(s.ljust(w) for s, w in zip(row_str, col_widths))
        print(line)


def get_summary_row(data: dict, log_path: str) -> tuple[list[str], list]:
    """返回摘要的一行数据：(表头列表, 值列表)，与控制台打印内容对应。"""
    n = len(data["Energy"]["iters"])
    last = n - 1
    E = data["Energy"]
    e_init = get_energy_mean(E, 0)
    e_final = get_energy_mean(E, last)
    sig_final = get_energy_sigma(E, last)
    acc = data.get("acceptance", {})
    v = acc.get("value", acc.get("Mean", [])) if acc else []
    a_init = v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else float("nan")
    a_final = v[last] if isinstance(v, (list, tuple)) and last < len(v) else (v[-1] if v else float("nan"))
    mx = get_obs_mean(data["Mx"], last) if "Mx" in data else float("nan")
    mz = get_obs_mean(data["Mz"], last) if "Mz" in data else float("nan")
    ntot = get_obs_mean(data["Ntot"], last) if "Ntot" in data else float("nan")
    headers = [
        "log_file", "n_iters",
        "E_initial", "E_final", "sigma_E_final",
        "accept_initial", "accept_final",
        "Mx_final", "Mz_final", "Ntot_final",
    ]
    values = [
        log_path, n,
        e_init, e_final, sig_final,
        a_init, a_final,
        mx, mz, ntot,
    ]
    return headers, values


def save_summary_csv(data: dict, log_path: str, csv_path: str) -> None:
    """将控制台打印的摘要信息写入 CSV（单行数据）。"""
    headers, values = get_summary_row(data, log_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerow(values)


def _get_next_run_number(base_dir: str, name: str) -> int:
    """在 base_dir 下查找已存在的 {name}_run*_parsed.csv，返回下一可用编号（第几次训练）。"""
    if not os.path.isdir(base_dir):
        return 1
    pattern = re.compile(rf"^{re.escape(name)}_run(\d+)_parsed\.csv$")
    max_run = 0
    for f in os.listdir(base_dir):
        m = pattern.match(f)
        if m:
            max_run = max(max_run, int(m.group(1)))
    return max_run + 1


def save_to_csv(data: dict, csv_path: str) -> None:
    """将全部迭代步数据写入 CSV 文件。"""
    n = len(data["Energy"]["iters"])
    steps = list(range(n))
    rows = get_table_rows(data, steps)
    csv_headers = ["iter", "Energy", "sigma_E", "Mx", "Mz", "Ntot", "accept"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_headers)
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="解析 NetKet VMC 的 .log 文件，输出人类可读摘要与表格。"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="log 文件路径（不指定时使用 train/<precision>/ 下默认文件名）",
    )
    parser.add_argument(
        "-p", "--precision",
        type=str,
        default="complex128",
        choices=("complex128", "complex64"),
        help="未指定 log 时使用的精度子目录（默认: complex128）",
    )
    parser.add_argument(
        "-t", "--table",
        action="store_true",
        help="额外打印首/中/末若干步的表格",
    )
    parser.add_argument(
        "-n", "--num-rows",
        type=int,
        default=15,
        help="表格显示时首尾各取多少步（默认 15），中间取 1 步",
    )
    parser.add_argument(
        "-o", "--csv",
        type=str,
        default=None,
        metavar="FILE",
        help="将全部迭代步数据保存为 CSV 文件；不指定时默认保存为与 log 同名的 _parsed.csv",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="不生成 CSV 文件，仅打印摘要",
    )
    parser.add_argument(
        "-r", "--run",
        type=int,
        default=None,
        metavar="N",
        help="第几次训练结果，用于文件名 name_runN_parsed.csv；不指定则自动递增，不覆盖已有文件",
    )
    args = parser.parse_args()

    if args.log_file:
        log_path = os.path.abspath(args.log_file)
    else:
        # 默认与 rydberg_nqs_starter.py 输出一致：rydberg_chain/train/<precision>/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, "train", args.precision, "rydberg_L16_delta0.5_Rb1.0_alpha6.log")

    if not os.path.isfile(log_path):
        print(f"错误: 未找到文件 {log_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_log(log_path)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON — {e}", file=sys.stderr)
        sys.exit(1)

    print_summary(data, log_path)

    if args.table:
        n = len(data["Energy"]["iters"])
        k = max(1, min(args.num_rows, n // 2))
        steps = list(range(0, k)) + [n // 2] + list(range(n - k, n))
        steps = sorted(set(steps))
        print("\n  【部分迭代步数据】\n")
        print_table(data, steps)

    # 保存 CSV（默认与 log 同目录，文件名带 run N 表示第几次训练，不覆盖已有）
    if not args.no_csv:
        if args.csv:
            csv_path = os.path.abspath(args.csv)
            base_dir = os.path.dirname(csv_path)
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            if base_name.endswith("_parsed"):
                summary_path = os.path.join(base_dir, base_name.replace("_parsed", "_summary") + ".csv")
            else:
                summary_path = os.path.join(base_dir, f"{base_name}_summary.csv")
        else:
            base_dir = os.path.dirname(os.path.abspath(log_path))
            name = os.path.splitext(os.path.basename(log_path))[0]
            run = args.run if args.run is not None else _get_next_run_number(base_dir, name)
            csv_path = os.path.join(base_dir, f"{name}_run{run}_parsed.csv")
            summary_path = os.path.join(base_dir, f"{name}_run{run}_summary.csv")
            print(f"\n  本次为第 {run} 次训练结果，保存为 *_run{run}_*.csv")
        save_to_csv(data, csv_path)
        save_summary_csv(data, log_path, summary_path)
        print(f"  迭代数据已保存: {csv_path}")
        print(f"  摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
