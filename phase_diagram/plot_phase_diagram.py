#!/usr/bin/env python3
"""
从所有参数点的 summary CSV 提取 |Mx| 和 |Mz_AFM|，绘制两张相图热力图。

横轴 J，纵轴 alphaInt，颜色映射分别为 |Mx|（横磁化）和 |Mz_AFM|（反铁磁序参量）。
同时输出汇总 CSV。
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


def _resolve_summary(base_dir, base_name):
    for prefix in ["_merged", "_run1", ""]:
        p = os.path.join(base_dir, f"{base_name}{prefix}_summary.csv")
        if os.path.isfile(p):
            return p
    return None


def _read_summary(path):
    """Read a summary CSV and return the last row as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return rows[-1]


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="绘制 |Mx| 与 |Mz_AFM| 相图")
    parser.add_argument("-p", "--precision", default=DEFAULT_PRECISION, choices=("complex64", "complex128"))
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--alpha-rbm", type=int, default=DEFAULT_ALPHA_RBM)
    parser.add_argument("--cal", type=int, default=DEFAULT_CAL)
    args = parser.parse_args()

    n_alpha = len(ALPHA_INT_LIST)
    n_j = len(J_LIST)
    mx_grid = np.full((n_alpha, n_j), np.nan)
    mz_afm_grid = np.full((n_alpha, n_j), np.nan)
    mz_afm_sq_grid = np.full((n_alpha, n_j), np.nan)
    e_grid = np.full((n_alpha, n_j), np.nan)

    summary_rows = []
    missing = 0

    for ai, alpha_int in enumerate(ALPHA_INT_LIST):
        for ji, J in enumerate(J_LIST):
            subdir = _param_subdir(args.L, J, args.delta, alpha_int)
            base_dir = os.path.join(_script_dir, "train", args.precision, subdir)
            base_name = _file_base(args.L, J, args.delta, alpha_int, args.alpha_rbm, args.cal)

            summary_path = _resolve_summary(base_dir, base_name)
            if summary_path is None:
                print(f"  MISSING: J={J}, alphaInt={alpha_int}")
                missing += 1
                continue

            row = _read_summary(summary_path)
            if row is None:
                missing += 1
                continue

            e_final = float(row.get("E_final", "nan"))
            mx_val = abs(float(row.get("Mx_final", "nan")))
            mz_afm_val = abs(float(row.get("Mz_AFM_final", "nan")))
            mz_afm_sq_val = float(row.get("Mz_AFM_sq_final", "nan"))

            mx_grid[ai, ji] = mx_val
            mz_afm_grid[ai, ji] = mz_afm_val
            mz_afm_sq_grid[ai, ji] = mz_afm_sq_val
            e_grid[ai, ji] = e_final

            summary_rows.append({
                "J": J, "alphaInt": alpha_int,
                "E_final": e_final,
                "abs_Mx": mx_val, "abs_Mz_AFM": mz_afm_val,
                "Mz_AFM_sq": mz_afm_sq_val,
            })

    print(f"Loaded {len(summary_rows)} / {n_alpha * n_j} points ({missing} missing)")

    # Save summary CSV
    fig_dir = os.path.join(_script_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    csv_out = os.path.join(fig_dir, f"phase_diagram_summary_L{args.L}.csv")
    if summary_rows:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["J", "alphaInt", "E_final", "abs_Mx", "abs_Mz_AFM", "Mz_AFM_sq"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary CSV: {csv_out}")

    if len(summary_rows) == 0:
        print("No data to plot.", file=sys.stderr)
        return

    # ---- Plotting ----
    J_arr = np.array(J_LIST)
    alpha_arr = np.array(ALPHA_INT_LIST)

    plt.style.use(["science", "no-latex"])

    for data_grid, obs_label, obs_tag in [
        (mx_grid, r"$|M_x|$", "Mx"),
        (mz_afm_grid, r"$|M_z^{\mathrm{AFM}}|$", "MzAFM"),
        (mz_afm_sq_grid, r"$\langle (M_z^{\mathrm{AFM}})^2 \rangle$", "MzAFM_sq"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.pcolormesh(
            J_arr, alpha_arr, data_grid,
            shading="nearest", cmap="viridis",
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(obs_label, fontsize=18)
        cb.ax.tick_params(labelsize=14)

        ax.set_xlabel(r"$J$", fontsize=20)
        ax.set_ylabel(r"$\alpha_{\mathrm{int}}$", fontsize=20)
        ax.set_xticks(J_arr)
        ax.set_yticks(alpha_arr)
        ax.tick_params(labelsize=14)
        ax.set_title(
            rf"Phase Diagram: {obs_label}, $L={args.L}$, $\delta={args.delta}$",
            fontsize=18,
        )

        # Annotate each cell with its value
        for ai in range(n_alpha):
            for ji in range(n_j):
                val = data_grid[ai, ji]
                if np.isfinite(val):
                    ax.text(J_arr[ji], alpha_arr[ai], f"{val:.3f}",
                            ha="center", va="center", fontsize=10,
                            color="white" if val < np.nanmedian(data_grid) else "black")

        plt.tight_layout()

        for ext in ("pdf", "svg"):
            path = os.path.join(fig_dir, f"PhaseDiagram_{obs_tag}_L{args.L}.{ext}")
            try:
                plt.savefig(path, dpi=150)
                print(f"  Saved: {path}")
            except (PermissionError, OSError) as e:
                print(f"  Save failed {path}: {e}", file=sys.stderr)
        plt.close(fig)


if __name__ == "__main__":
    main()
