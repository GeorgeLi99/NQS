#!/usr/bin/env python3
"""
为 phase_diagram 下每个参数点绘制 Energy 收敛 + 观测量图（两面板）。

照搬 long_range_ising/Fig_Convergence_Obs.py 的绘图逻辑，遍历所有
(J, alphaInt) 组合，PDF/SVG 保存在各自参数子目录内。
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    print("缺少 matplotlib/scienceplots，请 pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALPHA_INT_LIST, J_LIST,
    PRECISION as DEFAULT_PRECISION, L as DEFAULT_L, delta as DEFAULT_DELTA,
    alpha_rbm as DEFAULT_ALPHA_RBM, key_cal as DEFAULT_CAL,
)

_script_dir = os.path.dirname(os.path.abspath(__file__))


def _param_subdir(L, J, delta, alpha_int):
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def _file_base(L, J, delta, alpha_int, alpha_rbm, cal):
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{cal}"


# ======================================================================
# CSV 读取
# ======================================================================
def _read_parsed_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    cols = {}
    for k in rows[0].keys():
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except (ValueError, TypeError):
            pass
    return cols


def _resolve_csv(base_dir, base_name, suffix):
    for prefix in ["_merged", "_run1", ""]:
        p = os.path.join(base_dir, f"{base_name}{prefix}{suffix}.csv")
        if os.path.isfile(p):
            return p
    return os.path.join(base_dir, f"{base_name}_run1{suffix}.csv")


def load_E0(summary_path):
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return float(rows[-1]["E_final"])


# ======================================================================
# Plotting
# ======================================================================
def plot_one_point(parsed_path, summary_path, out_dir, L, J, delta, alpha_int):
    cols = _read_parsed_csv(parsed_path)
    iters = cols.get("global_iter", cols.get("iter"))
    energy = cols["Energy"]

    E0 = load_E0(summary_path)
    if E0 is None or E0 == 0:
        E0 = energy[-1]

    rel_err = np.abs((energy - E0) / E0)
    mx = cols.get("Mx")
    mz = cols.get("Mz")
    mz_afm = cols.get("Mz_AFM")
    sig_mx = cols.get("sigma_Mx")
    sig_mz = cols.get("sigma_Mz")
    sig_mz_afm = cols.get("sigma_Mz_AFM")

    colors = ["#085293", "#90d4bd", "#f58b47", "#fcce25", "#6300a7", "#a51f99"]

    plt.style.use(["science", "no-latex"])
    fig = plt.figure(figsize=(2 * 8.6, 6.45))

    # Left: energy convergence
    ax1 = plt.subplot(121)
    ax1.plot(iters, rel_err, color=colors[0], lw=2.0,
             label=rf"$J={J},\,\alpha_{{int}}={alpha_int}$")
    ax1.set_ylabel(r"$\epsilon = |{(E-E_0)}/{E_0}|$", fontsize=20)
    ax1.set_xlabel("Iteration", fontsize=20)
    ymax = float(np.nanmax(rel_err[np.isfinite(rel_err)])) if np.any(np.isfinite(rel_err)) else 1.0
    ax1.set_ylim((1e-8, max(1.0, ymax) * 1.05))
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=18, frameon=False)
    ax1.text(0.02, 0.02, rf"$E_0 = {E0:.6g}$", transform=ax1.transAxes, fontsize=14, verticalalignment="bottom")
    ax1.tick_params("both", which="major", length=4, direction="in")

    # Right: observables
    ax2 = plt.subplot(122)
    if mx is not None:
        ax2.plot(iters, np.abs(mx), color=colors[3], lw=2.0, label=r"$|M_x|$")
        if sig_mx is not None:
            ax2.fill_between(iters, np.abs(mx) - sig_mx, np.abs(mx) + sig_mx, color=colors[3], alpha=0.2)
    if mz is not None:
        ax2.plot(iters, np.abs(mz), color=colors[4], lw=2.0, label=r"$|M_z|$")
        if sig_mz is not None:
            ax2.fill_between(iters, np.abs(mz) - sig_mz, np.abs(mz) + sig_mz, color=colors[4], alpha=0.2)
    if mz_afm is not None:
        ax2.plot(iters, np.abs(mz_afm), color=colors[5], lw=2.0, label=r"$|M_z^{\mathrm{AFM}}|$")
        if sig_mz_afm is not None:
            ax2.fill_between(iters, np.abs(mz_afm) - sig_mz_afm, np.abs(mz_afm) + sig_mz_afm, color=colors[5], alpha=0.2)
    ax2.set_ylabel(r"$M_x,\, M_z,\, M_z^{\mathrm{AFM}}$", fontsize=20)
    ax2.set_xlabel("Iteration", fontsize=20)
    ax2.legend(loc="upper right", fontsize=18, frameon=False)
    ax2.tick_params("both", which="major", length=4, direction="in")

    plt.suptitle(
        rf"Long-range Ising: $J={J},\,\delta={delta},\,L={L},\,\alpha_{{int}}={alpha_int}$",
        fontsize=20,
    )
    plt.tight_layout()

    tag = f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"
    for ext in ("pdf", "svg"):
        path = os.path.join(out_dir, f"Fig_ConvObs_{tag}.{ext}")
        try:
            plt.savefig(path)
            print(f"    saved: {path}")
        except (PermissionError, OSError) as e:
            print(f"    save failed {path}: {e}", file=sys.stderr)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="为每个相图采样点绘制收敛+观测量图")
    parser.add_argument("-p", "--precision", default=DEFAULT_PRECISION, choices=("complex64", "complex128"))
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--alpha-rbm", type=int, default=DEFAULT_ALPHA_RBM)
    parser.add_argument("--cal", type=int, default=DEFAULT_CAL)
    args = parser.parse_args()

    ok, skip = 0, 0
    for alpha_int in ALPHA_INT_LIST:
        for J in J_LIST:
            subdir = _param_subdir(args.L, J, args.delta, alpha_int)
            base_dir = os.path.join(_script_dir, "train", args.precision, subdir)
            base_name = _file_base(args.L, J, args.delta, alpha_int, args.alpha_rbm, args.cal)

            parsed_path = _resolve_csv(base_dir, base_name, "_parsed")
            summary_path = _resolve_csv(base_dir, base_name, "_summary")

            if not os.path.isfile(parsed_path):
                print(f"  SKIP (no CSV): J={J}, alphaInt={alpha_int}")
                skip += 1
                continue

            print(f"  Plotting J={J}, alphaInt={alpha_int}")
            plot_one_point(parsed_path, summary_path, base_dir, args.L, J, args.delta, alpha_int)
            ok += 1

    print(f"\nDone: {ok} plotted, {skip} skipped")


if __name__ == "__main__":
    main()
