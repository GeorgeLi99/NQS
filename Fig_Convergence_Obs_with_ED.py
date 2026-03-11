#!/usr/bin/env python3
"""
绘制 long_range_ising 的 VMC 收敛与观测量，但将参考能量 E0 替换为 ED（QuSpin Lanczos）结果。

ED 结果读取自：ED/result/ising_L{L}_J{J}_alpha{alpha}_delta{delta}_h{h}.csv
VMC 数据读取自：long_range_ising/train/<precision>/L{L}_J{J}_delta{delta}_alphaInt{alphaInt}/ 下的 *_merged_parsed.csv（或 run1/parsed 备选）

用法示例：
  python3 Fig_Convergence_Obs_with_ED.py -p complex64 --L 16 --J 1.0 --delta 0.0 --alpha-int 2.0
"""

import argparse
import csv
import os
import sys

try:
    import numpy as np
except ModuleNotFoundError:
    print("缺少 numpy，请 pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

try:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    print("缺少 matplotlib/scienceplots，请 pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)


_root_dir = os.path.dirname(os.path.abspath(__file__))
_ising_dir = os.path.join(_root_dir, "long_range_ising")
_ed_result_dir = os.path.join(_root_dir, "ED", "result")


# --- 默认超参数（可由命令行覆盖）---
DEFAULT_PRECISION = "complex64"
DEFAULT_L = 16
DEFAULT_J = 1.0
DEFAULT_DELTA = 0.5
DEFAULT_ALPHA_INT = 6.0

# ED 的参数：默认认为 alpha(ED) == alphaInt，且 h = Omega/2 = 1.0
DEFAULT_ED_ALPHA = None  # None -> 使用 alphaInt
DEFAULT_ED_H = 1.0


def _param_subdir_from_params(L: int, J: float, delta: float, alpha_int: float) -> str:
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def _basename_from_params(L: int, J: float, delta: float, alpha_int: float, alpha_rbm: int = 4, cal: int = 1) -> str:
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{cal}"


def _resolve_csv(base_dir: str, base_name: str, suffix: str) -> str:
    for prefix in ["_merged", "_run1", ""]:
        p = os.path.join(base_dir, f"{base_name}{prefix}{suffix}.csv")
        if os.path.isfile(p):
            return p
    return os.path.join(base_dir, f"{base_name}_merged{suffix}.csv")


def _read_ed_e0(L: int, J: float, alpha_ed: float, delta: float, h: float) -> float:
    path = os.path.join(_ed_result_dir, f"ising_L{L}_J{J}_alpha{alpha_ed}_delta{delta}_h{h}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 ED CSV：{path}\n"
            f"请先运行：python3 ED/ground_state_ising_lanczos.py（并确保参数一致）"
        )
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"ED CSV 为空：{path}")
    return float(rows[-1]["gs_energy"])


def _read_parsed_csv(parsed_path: str) -> dict[str, np.ndarray]:
    with open(parsed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {parsed_path}")
    cols: dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except (ValueError, TypeError):
            # 非数值列忽略
            pass
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="用 ED 的 E0 绘制 long_range_ising 收敛与观测量")
    parser.add_argument("--precision", "-p", choices=("complex64", "complex128"), default=DEFAULT_PRECISION)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--J", type=float, default=DEFAULT_J)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--alpha-int", type=float, default=DEFAULT_ALPHA_INT, dest="alpha_int")
    parser.add_argument("--alpha-rbm", type=int, default=4, help="RBM hidden alpha（文件名用）")
    parser.add_argument("--cal", type=int, default=1, help="Cal key（文件名用）")
    parser.add_argument("--ed-alpha", type=float, default=DEFAULT_ED_ALPHA,
                        help="ED 的相互作用指数 alpha；默认等于 alpha-int")
    parser.add_argument("--ed-h", type=float, default=DEFAULT_ED_H,
                        help="ED 的横场系数 h（对应 Ω/2）；默认 1.0")
    args = parser.parse_args()

    alpha_ed = args.alpha_int if args.ed_alpha is None else args.ed_alpha
    E0 = _read_ed_e0(args.L, args.J, alpha_ed, args.delta, args.ed_h)

    param_subdir = _param_subdir_from_params(args.L, args.J, args.delta, args.alpha_int)
    data_dir = os.path.join(_ising_dir, "train", args.precision, param_subdir)
    base_name = _basename_from_params(args.L, args.J, args.delta, args.alpha_int, args.alpha_rbm, args.cal)
    parsed_csv = _resolve_csv(data_dir, base_name, "_parsed")

    if not os.path.isfile(parsed_csv):
        raise FileNotFoundError(
            f"找不到 VMC parsed CSV：{parsed_csv}\n"
            f"请先在对应目录运行 long_range_ising 的 parse/merge 脚本。"
        )

    cols = _read_parsed_csv(parsed_csv)
    iters = cols.get("global_iter", cols.get("iter"))
    if iters is None:
        iters = np.arange(len(cols["Energy"]), dtype=np.float64)
    energy = cols["Energy"]
    rel_err = np.abs((energy - E0) / E0)

    mx = cols.get("Mx")
    mz = cols.get("Mz")
    mz_afm = cols.get("Mz_AFM")
    sig_mx = cols.get("sigma_Mx")
    sig_mz = cols.get("sigma_Mz")
    sig_mz_afm = cols.get("sigma_Mz_AFM")

    colors = ["#085293", "#90d4bd", "#f58b47", "#fcce25", "#6300a7", "#a51f99", "#b7ea63"]

    plt.style.use(["science", "no-latex"])
    plt.figure(figsize=(2 * 8.6, 6.45))

    # Left: energy convergence (ED reference)
    ax1 = plt.subplot(121)
    ax1.plot(iters, rel_err, color=colors[0], lw=2.0,
             label=rf"$\delta={args.delta},\, \alpha_{{int}}=$" + f" {args.alpha_int}")
    ax1.set_ylabel(r'$\epsilon = |\frac{E-E_0}{E_0}|$', fontsize=25)
    ax1.set_xlabel("Iteration", fontsize=25)
    ax1.set_yscale("log")
    ymax = float(np.nanmax(rel_err[np.isfinite(rel_err)])) if np.any(np.isfinite(rel_err)) else 1.0
    ax1.set_ylim((1e-8, max(1.0, ymax) * 1.05))
    ax1.legend(loc="upper right", fontsize=25, frameon=False)
    ax1.text(0.02, 0.02, rf"$E_0^{{ED}} = {E0:.6g}$", transform=ax1.transAxes, fontsize=18, verticalalignment="bottom")
    ax1.tick_params("both", which="major", length=4, direction="in")
    ax1.tick_params("both", which="minor", length=2, direction="in")

    # Right: observables
    ax2 = plt.subplot(122)
    if mx is not None:
        mx_arr = np.abs(np.asarray(mx))
        ax2.plot(iters, mx_arr, color=colors[3], lw=2.0, label=rf"$|M_x|$")
        if sig_mx is not None:
            ax2.fill_between(iters, mx_arr - sig_mx, mx_arr + sig_mx, color=colors[3], alpha=0.2)
    if mz is not None:
        mz_arr = np.abs(np.asarray(mz))
        ax2.plot(iters, mz_arr, color=colors[4], lw=2.0, label=rf"$|M_z|$")
        if sig_mz is not None:
            ax2.fill_between(iters, mz_arr - sig_mz, mz_arr + sig_mz, color=colors[4], alpha=0.2)
    if mz_afm is not None:
        mz_afm_arr = np.abs(np.asarray(mz_afm))
        ax2.plot(iters, mz_afm_arr, color=colors[5], lw=2.0, label=rf"$|M_z^{{AFM}}|$")
        if sig_mz_afm is not None:
            ax2.fill_between(iters, mz_afm_arr - sig_mz_afm, mz_afm_arr + sig_mz_afm, color=colors[5], alpha=0.2)

    ax2.set_ylabel(r'$M_x,\, M_z,\, M_z^{\mathrm{AFM}}$', fontsize=25)
    ax2.set_xlabel("Iteration", fontsize=25)
    ax2.legend(loc="upper right", fontsize=25, frameon=False)
    ax2.tick_params("both", which="major", length=4, direction="in")
    ax2.tick_params("both", which="minor", length=2, direction="in")

    plt.suptitle(
        r"Long-range Ising (ED ref): "
        + rf"$\delta={args.delta},\,L={args.L},\,J={args.J},\,\alpha_{{int}}={args.alpha_int}$",
        fontsize=22,
    )
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.tight_layout()

    out_dir = os.path.join(_root_dir, "ED_vs_NQS_figure")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"L{args.L}_J{args.J}_delta{args.delta}_alphaInt{args.alpha_int}_EDalpha{alpha_ed}_h{args.ed_h}"
    for ext in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"Fig_ConvObs_RBM_LongIsing_EDref_{tag}.{ext}")
        plt.savefig(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

