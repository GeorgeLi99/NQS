#!/usr/bin/env python3
"""
批量解析 phase_diagram/train/<precision>/ 下所有参数点的 .log 文件。

对每个 (J, alphaInt) 子目录，调用与 long_range_ising/parse_vmc_log.py 相同的
解析逻辑，生成 *_run1_parsed.csv 和 *_run1_summary.csv。
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALPHA_INT_LIST, J_LIST,
    PRECISION as DEFAULT_PRECISION, L as DEFAULT_L, delta as DEFAULT_DELTA,
    alpha_rbm as DEFAULT_ALPHA_RBM, key_cal as DEFAULT_CAL,
)


def _param_subdir(L, J, delta, alpha_int):
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def _file_base(L, J, delta, alpha_int, alpha_rbm, cal):
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{cal}"


# ======================================================================
# Log 解析函数（照搬 long_range_ising/parse_vmc_log.py）
# ======================================================================
_OBS_COLS = ["Mx", "Mz", "Mz_AFM", "Ntot"]


def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_energy_mean(entry: dict, i: int) -> float:
    mean = entry.get("Mean")
    if isinstance(mean, dict) and "real" in mean:
        return mean["real"][i]
    if isinstance(mean, (list, tuple)):
        return mean[i] if i < len(mean) else float("nan")
    return float("nan")


def get_energy_sigma(entry: dict, i: int) -> float:
    sigma = entry.get("Sigma", [])
    if not isinstance(sigma, (list, tuple)):
        return float("nan")
    if i >= len(sigma):
        return float("nan")
    return float("nan") if sigma[i] is None else sigma[i]


def get_obs_mean(entry: dict, i: int) -> float:
    mean = entry.get("Mean")
    if isinstance(mean, dict) and "real" in mean:
        arr = mean["real"]
        return arr[i] if i < len(arr) else float("nan")
    if isinstance(mean, (list, tuple)):
        return mean[i] if i < len(mean) else float("nan")
    return float("nan")


def get_obs_sigma(entry: dict, i: int) -> float:
    sigma = entry.get("Sigma", [])
    if not isinstance(sigma, (list, tuple)):
        return float("nan")
    if i >= len(sigma):
        return float("nan")
    return float("nan") if sigma[i] is None else sigma[i]


def get_acceptance(entry: dict, i: int) -> float:
    val = entry.get("value", entry.get("Mean", []))
    if isinstance(val, (list, tuple)):
        return val[i] if i < len(val) else float("nan")
    return float("nan")


def _csv_headers() -> list[str]:
    h = ["iter", "Energy", "sigma_E"]
    for name in _OBS_COLS:
        h.append(name)
        h.append(f"sigma_{name}")
    h.append("accept")
    return h


def get_table_rows(data: dict, steps: list) -> list[list]:
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


def save_to_csv(data: dict, csv_path: str) -> None:
    n = len(data["Energy"]["iters"])
    rows = get_table_rows(data, list(range(n)))
    headers = _csv_headers()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def save_summary_csv(data: dict, log_path: str, csv_path: str) -> None:
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
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerow(values)


def _get_next_run_number(base_dir: str, name: str) -> int:
    if not os.path.isdir(base_dir):
        return 1
    pattern = re.compile(rf"^{re.escape(name)}_run(\d+)_parsed\.csv$")
    max_run = 0
    for f in os.listdir(base_dir):
        m = pattern.match(f)
        if m:
            max_run = max(max_run, int(m.group(1)))
    return max_run + 1


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="批量解析 phase_diagram 下所有参数点的 .log 文件")
    parser.add_argument("-p", "--precision", default=DEFAULT_PRECISION, choices=("complex64", "complex128"))
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--alpha-rbm", type=int, default=DEFAULT_ALPHA_RBM)
    parser.add_argument("--cal", type=int, default=DEFAULT_CAL)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ok, skip, fail = 0, 0, 0

    for alpha_int in ALPHA_INT_LIST:
        for J in J_LIST:
            subdir = _param_subdir(args.L, J, args.delta, alpha_int)
            base_dir = os.path.join(script_dir, "train", args.precision, subdir)
            base_name = _file_base(args.L, J, args.delta, alpha_int, args.alpha_rbm, args.cal)
            log_path = os.path.join(base_dir, base_name + ".log")

            if not os.path.isfile(log_path):
                print(f"  SKIP (no log): {log_path}")
                skip += 1
                continue

            try:
                data = load_log(log_path)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  FAIL: {log_path}: {e}", file=sys.stderr)
                fail += 1
                continue

            run = _get_next_run_number(base_dir, base_name)
            csv_path = os.path.join(base_dir, f"{base_name}_run{run}_parsed.csv")
            summary_path = os.path.join(base_dir, f"{base_name}_run{run}_summary.csv")

            save_to_csv(data, csv_path)
            save_summary_csv(data, log_path, summary_path)
            print(f"  OK  J={J}, alphaInt={alpha_int} -> run{run}")
            ok += 1

    print(f"\nDone: {ok} parsed, {skip} skipped, {fail} failed (total {ok + skip + fail})")


if __name__ == "__main__":
    main()
