#!/usr/bin/env python3
"""
解析 NetKet VMC 生成的 .log 文件，输出便于阅读的摘要与表格，并可将结果保存为 CSV。
（与 rydberg_chain 相同逻辑，默认路径与基名为 long_range_ising/train/<precision>/ 下的 rbm_LongIsing_*）

用法:
  python3 parse_vmc_log.py [log 文件路径]
  python3 parse_vmc_log.py   # 默认解析 train/<precision>/rbm_LongIsing_L=16_J=1.0_delta=0.0_... .log
  python3 parse_vmc_log.py --delta 0 --precision complex64
  python3 parse_vmc_log.py --delta 0.5   # 处理 delta=0.5 的 log

输出:
  - 运行概览（迭代数、最终能量与误差、接受率等）
  - 观测量摘要（Mx, Mz；本模型无 Ntot）
  - 可选：指定步数的能量与观测量表格
  - 默认保存为 name_runN_parsed.csv / name_runN_summary.csv
"""

import argparse
import csv
import json
import os
import re
import sys

# --- 默认超参数（可由命令行覆盖）---
DEFAULT_PRECISION = "complex64"
DEFAULT_L = 16
DEFAULT_J = 1.0
DEFAULT_DELTA = 0.0
DEFAULT_ALPHA = 2.0

# 与 rbm_long_range_ising.py 输出一致；路径为 train/<precision>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/
def _param_subdir_from_params(L: int, J: float, delta: float, alpha_int: float) -> str:
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def _basename_from_params(L: int, J: float, delta: float, alpha_int: float) -> str:
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha=4_Cal1"


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
    """从观测量条目（Mx/Mz/Mz_AFM/Ntot）中取第 i 步的 Mean（实部或标量）。"""
    mean = entry.get("Mean")
    if isinstance(mean, dict) and "real" in mean:
        arr = mean["real"]
        return arr[i] if i < len(arr) else float("nan")
    if isinstance(mean, (list, tuple)):
        return mean[i] if i < len(mean) else float("nan")
    return float("nan")


def get_obs_sigma(entry: dict, i: int) -> float:
    """从观测量条目中取第 i 步的 Sigma（误差/标准差），用于看 variance 量级。"""
    sigma = entry.get("Sigma", [])
    return sigma[i] if i < len(sigma) else float("nan")


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
    print("  VMC 运行日志解析结果 (Long-range Ising)")
    print("=" * 60)
    print(f"  文件: {log_path}")
    print(f"  总迭代步数: {n}")
    print()

    E = data["Energy"]
    e_final = get_energy_mean(E, last)
    sig_final = get_energy_sigma(E, last)
    e_init = get_energy_mean(E, 0)
    print("  【能量 E = ⟨ψ|H|ψ⟩】")
    print(f"    初始 (iter=0):   {format_number(e_init)}")
    print(f"    最终 (iter={last}): {format_number(e_final)}  ±  {format_number(sig_final)}")
    print()

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

    obs_names = ["Mx", "Mz", "Mz_AFM", "Ntot"]
    obs_desc = {
        "Mx": "横磁化 (1/L)Σ⟨σˣ⟩",
        "Mz": "纵磁化 (1/L)Σ⟨σᶻ⟩",
        "Mz_AFM": "反铁磁序参量 (1/L)Σ(-1)ʲ⟨σᶻⱼ⟩",
        "Ntot": "占据数 (1/L)Σ⟨n⟩",
    }
    print("  【观测量（最终步，含 Sigma 便于看 variance）】")
    for name in obs_names:
        if name not in data:
            continue
        val = get_obs_mean(data[name], last)
        sig = get_obs_sigma(data[name], last)
        desc = obs_desc.get(name, name)
        print(f"    {name} ({desc}): {format_number(val)}  ±  {format_number(sig)}")
    print("=" * 60)


# 观测量列表（与 rbm_long_range_ising 中 obs 一致；Ntot 可选）
_OBS_COLS = ["Mx", "Mz", "Mz_AFM", "Ntot"]


def get_table_rows(data: dict, steps: list) -> list[list]:
    """返回指定步数的数据行。含 Energy, sigma_E，各观测量 mean/sigma，accept。"""
    E = data["Energy"]
    acc = data.get("acceptance", {})
    rows = []
    for i in steps:
        if i >= len(E["iters"]):
            continue
        e_val = get_energy_mean(E, i)
        e_sig = get_energy_sigma(E, i)
        row = [i, e_val, e_sig]
        for name in _OBS_COLS:
            if name in data:
                row.append(get_obs_mean(data[name], i))
                row.append(get_obs_sigma(data[name], i))
            else:
                row.append(float("nan"))
                row.append(float("nan"))
        a = get_acceptance(acc, i) if acc else float("nan")
        row.append(a)
        rows.append(row)
    return rows


def _csv_headers() -> list[str]:
    """解析/CSV 表头：iter, Energy, sigma_E, 各观测量 mean/sigma, accept。"""
    h = ["iter", "Energy", "sigma_E"]
    for name in _OBS_COLS:
        h.append(name)
        h.append(f"sigma_{name}")
    h.append("accept")
    return h


def print_table(data: dict, steps: list) -> None:
    """打印指定步数的能量与观测量表格。"""
    rows = get_table_rows(data, steps)
    headers = _csv_headers()
    col_widths = [6, 14, 10] + [12, 10] * len(_OBS_COLS) + [8]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        row_str = [str(r[0]), format_number(r[1], 5), format_number(r[2], 4)]
        idx = 3
        for _ in _OBS_COLS:
            row_str.append(format_number(r[idx], 5))
            row_str.append(format_number(r[idx + 1], 4))
            idx += 2
        row_str.append(format_number(r[-1], 3))
        line = "  ".join(s.ljust(w) for s, w in zip(row_str, col_widths))
        print(line)


def get_summary_row(data: dict, log_path: str) -> tuple[list[str], list]:
    """返回摘要的一行数据：(表头列表, 值列表)。含各观测量 final 与 sigma。"""
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
    headers = [
        "log_file", "n_iters",
        "E_initial", "E_final", "sigma_E_final",
        "accept_initial", "accept_final",
    ]
    values = [log_path, n, e_init, e_final, sig_final, a_init, a_final]
    for name in _OBS_COLS:
        if name in data:
            headers.append(f"{name}_final")
            headers.append(f"sigma_{name}_final")
            values.append(get_obs_mean(data[name], last))
            values.append(get_obs_sigma(data[name], last))
        else:
            headers.append(f"{name}_final")
            headers.append(f"sigma_{name}_final")
            values.append(float("nan"))
            values.append(float("nan"))
    return headers, values


def save_summary_csv(data: dict, log_path: str, csv_path: str) -> None:
    """将控制台打印的摘要信息写入 CSV（单行数据）。"""
    headers, values = get_summary_row(data, log_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerow(values)


def _get_next_run_number(base_dir: str, name: str) -> int:
    """在 base_dir 下查找已存在的 {name}_run*_parsed.csv，返回下一可用编号。"""
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
    """将全部迭代步数据写入 CSV 文件（含各观测量 mean 与 sigma）。"""
    n = len(data["Energy"]["iters"])
    steps = list(range(n))
    rows = get_table_rows(data, steps)
    csv_headers = _csv_headers()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_headers)
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="解析 NetKet VMC 的 .log 文件（Long-range Ising），输出人类可读摘要与表格。"
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
        default=DEFAULT_PRECISION,
        choices=("complex128", "complex64"),
        help=f"未指定 log 时使用的精度子目录（默认: {DEFAULT_PRECISION}）",
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
        help="表格显示时首尾各取多少步（默认 15）",
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
        help="第几次训练结果，用于文件名 name_runN_parsed.csv",
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L, help=f"链长（默认: {DEFAULT_L}）")
    parser.add_argument("--J", type=float, default=DEFAULT_J, help=f"耦合强度（默认: {DEFAULT_J}）")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help=f"纵场 detuning（默认: {DEFAULT_DELTA}）")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"相互作用指数 α（默认: {DEFAULT_ALPHA}）")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="log/CSV 基名；不指定时由 L/J/delta/alpha 构造",
    )
    args = parser.parse_args()

    param_subdir = _param_subdir_from_params(args.L, args.J, args.delta, args.alpha)
    base_name = args.name if args.name is not None else _basename_from_params(args.L, args.J, args.delta, args.alpha)

    if args.log_file:
        log_path = os.path.abspath(args.log_file)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, "train", args.precision, param_subdir, base_name + ".log")

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
            base_name = os.path.splitext(os.path.basename(log_path))[0]
            run = args.run if args.run is not None else _get_next_run_number(base_dir, base_name)
            csv_path = os.path.join(base_dir, f"{base_name}_run{run}_parsed.csv")
            summary_path = os.path.join(base_dir, f"{base_name}_run{run}_summary.csv")
            print(f"\n  本次为第 {run} 次训练结果，保存为 *_run{run}_*.csv")
        save_to_csv(data, csv_path)
        save_summary_csv(data, log_path, summary_path)
        print(f"  迭代数据已保存: {csv_path}")
        print(f"  摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
