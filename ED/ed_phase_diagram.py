#!/usr/bin/env python3
"""
用 ED (Lanczos) 遍历 (J, alpha) 网格计算基态能量与序参量，绘制相图。

哈密顿量与 phase_diagram/run_phase_diagram.py 一致：
  H = (Ω/2) Σ_j σ^x_j  -  δ Σ_j σ^z_j  +  Σ_{i<j} (J/r^α) σ^z_i σ^z_j
  r 为周期边界下最短距离。

参数网格从 phase_diagram/config.py 读取（ALPHA_INT_LIST, J_LIST, L, delta, Omega）。
输出：
  - ED/result/phase_diagram_L{L}.csv（汇总表）
  - ED/figure/PhaseDiagram_ED_{obs}_L{L}.{pdf,svg}
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

# 读取 phase_diagram/config.py 中的参数网格
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase_diagram"))
from config import ALPHA_INT_LIST, J_LIST, L as DEFAULT_L, delta as DEFAULT_DELTA, Omega as DEFAULT_OMEGA


def compute_ground_state(L: int, J: float, alpha: float, delta: float, Omega: float):
    """
    用 QuSpin Lanczos 求基态。
    返回 (gs_energy, mx, mz, mz_afm)。
    """
    basis = spin_basis_1d(L, S="1/2")

    # σ^z σ^z 长程相互作用
    Jz_list = []
    for i in range(L):
        for j in range(i + 1, L):
            r = min(abs(i - j), L - abs(i - j))
            Jz_list.append([J / r ** alpha, i, j])

    # 横场 (Ω/2) σ^x
    hx_list = [[Omega / 2, i] for i in range(L)]

    # 纵场 -δ σ^z
    hz_list = [[-delta, i] for i in range(L)]

    static = [["zz", Jz_list], ["x", hx_list], ["z", hz_list]]
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128, check_symm=False, check_herm=False)

    E, V = H.eigsh(k=1, which="SA")
    gs_energy = E[0]
    gs = V[:, 0]

    # Mx = (1/L) Σ <σ^x_j>
    Sx_ops = [hamiltonian([["x", [[1, i]]]], [], basis=basis, dtype=np.complex128,
                          check_symm=False, check_herm=False) for i in range(L)]
    mx = sum(Sx_ops[i].expt_value(gs).real for i in range(L)) / L

    # Mz = (1/L) Σ <σ^z_j>
    Sz_ops = [hamiltonian([["z", [[1, i]]]], [], basis=basis, dtype=np.complex128,
                          check_symm=False, check_herm=False) for i in range(L)]
    mz_vals = [Sz_ops[i].expt_value(gs).real for i in range(L)]
    mz = sum(mz_vals) / L

    # Mz_AFM = (1/L) Σ (-1)^j <σ^z_j>
    mz_afm = sum((-1) ** j * mz_vals[j] for j in range(L)) / L

    return gs_energy, mx, mz, mz_afm


def main():
    parser = argparse.ArgumentParser(description="ED 相图：遍历 (J, alpha) 网格")
    parser.add_argument("--L", type=int, default=DEFAULT_L, help=f"链长（默认: {DEFAULT_L}）")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help=f"纵场（默认: {DEFAULT_DELTA}）")
    parser.add_argument("--Omega", type=float, default=DEFAULT_OMEGA, help=f"横场系数（默认: {DEFAULT_OMEGA}）")
    args = parser.parse_args()

    L = args.L
    delta_val = args.delta
    Omega_val = args.Omega

    alpha_list = sorted(ALPHA_INT_LIST)
    j_list = sorted(J_LIST)

    n_alpha = len(alpha_list)
    n_j = len(j_list)
    total = n_alpha * n_j

    print(f"ED Phase Diagram: L={L}, delta={delta_val}, Omega={Omega_val}")
    print(f"  ALPHA_INT_LIST = {alpha_list}")
    print(f"  J_LIST         = {j_list}")
    print(f"  Total points: {total}")
    print()

    # 存储结果
    rows = []
    e_grid = np.full((n_alpha, n_j), np.nan)
    mx_grid = np.full((n_alpha, n_j), np.nan)
    mz_grid = np.full((n_alpha, n_j), np.nan)
    mz_afm_grid = np.full((n_alpha, n_j), np.nan)

    total_start = time.time()
    for ai, alpha_int in enumerate(alpha_list):
        for ji, J_val in enumerate(j_list):
            t0 = time.time()
            print(f"  [{ai * n_j + ji + 1}/{total}]  J={J_val}, alpha={alpha_int} ...", end="", flush=True)

            gs_energy, mx, mz, mz_afm = compute_ground_state(L, J_val, alpha_int, delta_val, Omega_val)

            elapsed = time.time() - t0
            print(f"  E={gs_energy:.10f}  |Mx|={abs(mx):.6f}  |Mz_AFM|={abs(mz_afm):.6f}  ({elapsed:.1f}s)")

            e_grid[ai, ji] = gs_energy
            mx_grid[ai, ji] = abs(mx)
            mz_grid[ai, ji] = abs(mz)
            mz_afm_grid[ai, ji] = abs(mz_afm)

            rows.append({
                "J": J_val, "alphaInt": alpha_int,
                "E_gs": gs_energy,
                "Mx": mx, "abs_Mx": abs(mx),
                "Mz": mz, "abs_Mz": abs(mz),
                "Mz_AFM": mz_afm, "abs_Mz_AFM": abs(mz_afm),
            })

    total_elapsed = time.time() - total_start
    print(f"\nAll {total} points finished in {total_elapsed:.1f}s")

    # ---- 保存 CSV ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, f"phase_diagram_ED_L{L}.csv")
    fieldnames = ["J", "alphaInt", "E_gs", "Mx", "abs_Mx", "Mz", "abs_Mz", "Mz_AFM", "abs_Mz_AFM"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Summary CSV: {csv_path}")

    # ---- 绘图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401
    except ModuleNotFoundError:
        print("matplotlib/scienceplots 不可用，跳过绘图。")
        return

    fig_dir = os.path.join(base_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)

    J_arr = np.array(j_list)
    alpha_arr = np.array(alpha_list)

    plt.style.use(["science", "no-latex"])

    plot_specs = [
        (mx_grid, r"$|M_x|$", "Mx"),
        (mz_afm_grid, r"$|M_z^{\mathrm{AFM}}|$", "MzAFM"),
        (e_grid, r"$E_{\mathrm{gs}}$", "Energy"),
    ]

    for data_grid, obs_label, obs_tag in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.pcolormesh(J_arr, alpha_arr, data_grid, shading="nearest", cmap="viridis")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(obs_label, fontsize=18)
        cb.ax.tick_params(labelsize=14)

        ax.set_xlabel(r"$J$", fontsize=20)
        ax.set_ylabel(r"$\alpha_{\mathrm{int}}$", fontsize=20)
        ax.set_xticks(J_arr)
        ax.set_yticks(alpha_arr)
        ax.tick_params(labelsize=14)
        ax.set_title(
            rf"ED Phase Diagram: {obs_label}, $L={L}$, $\delta={delta_val}$",
            fontsize=18,
        )

        for ai in range(n_alpha):
            for ji in range(n_j):
                val = data_grid[ai, ji]
                if np.isfinite(val):
                    fmt = f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}"
                    ax.text(J_arr[ji], alpha_arr[ai], fmt,
                            ha="center", va="center", fontsize=8,
                            color="white" if val < np.nanmedian(data_grid) else "black")

        plt.tight_layout()

        for ext in ("pdf", "svg"):
            path = os.path.join(fig_dir, f"PhaseDiagram_ED_{obs_tag}_L{L}.{ext}")
            try:
                plt.savefig(path, dpi=150)
                print(f"  Saved: {path}")
            except (PermissionError, OSError) as e:
                print(f"  Save failed: {e}", file=sys.stderr)
        plt.close(fig)


if __name__ == "__main__":
    main()
