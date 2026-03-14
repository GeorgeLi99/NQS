#!/usr/bin/env python3
"""
DMRG vs NQS 能量比对相图
========================
解析 DMRG/L=32_delta=0.0_DMRG_energy.0_dmrg_energy_tmp 得到各 (J, alpha) 的基态能量，
同时从 phase_diagram/train/complex64/ 读取相同参数的 NQS 最终能量，
绘制三张相图：
  1. DMRG 能量热力图
  2. NQS 能量热力图
  3. 相对误差 (E_NQS - E_DMRG) / |E_DMRG|  热力图

输出保存到 ED_vs_NQS_figure/ 目录。
"""

from __future__ import annotations

import csv
import os
import re
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ModuleNotFoundError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ======================================================================
# 路径
# ======================================================================
_root = os.path.dirname(os.path.abspath(__file__))
DMRG_FILE = os.path.join(_root, "DMRG", "L=32_delta=0.0_DMRG_energy.0_dmrg_energy_tmp")
NQS_TRAIN_DIR = os.path.join(_root, "phase_diagram", "train", "complex64")
FIG_DIR = os.path.join(_root, "DMRG_vs_NQS_figure")
os.makedirs(FIG_DIR, exist_ok=True)

# ======================================================================
# 解析 DMRG 结果文件
# ======================================================================
def parse_dmrg(path: str) -> dict[tuple[float, float], float]:
    """
    返回 {(J, alpha_int): energy} 字典。
    每组 (J, alpha_int) 取文件中最后出现的 "After sweep N energy=..." 那一行。
    """
    result: dict[tuple[float, float], float] = {}

    current_J: float | None = None
    current_alpha: float | None = None
    current_energy: float | None = None

    # 用于识别 J 段标题：### {L = 32, delta = 0} J = X.X
    pat_J = re.compile(r"J\s*=\s*([\d.]+)", re.IGNORECASE)
    # alpha_int = X.X
    pat_alpha = re.compile(r"alpha_int\s*=\s*([\d.]+)")
    # After sweep N energy=...（能量数字中间可能有空格，要去掉）
    pat_energy = re.compile(r"After sweep\s+\d+\s+energy=\s*([\d.\-Ee \+]+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 标题行：### ... J = X.X
            if line.startswith("###"):
                # 保存上一组
                if current_J is not None and current_alpha is not None and current_energy is not None:
                    result[(current_J, current_alpha)] = current_energy
                m = pat_J.search(line)
                if m:
                    current_J = float(m.group(1))
                current_alpha = None
                current_energy = None
                continue

            # alpha_int 行
            m = pat_alpha.match(line)
            if m:
                # 保存上一组 alpha
                if current_J is not None and current_alpha is not None and current_energy is not None:
                    result[(current_J, current_alpha)] = current_energy
                current_alpha = float(m.group(1))
                current_energy = None
                continue

            # energy 行
            m = pat_energy.search(line)
            if m:
                # 去掉数字中的空格（处理 "49.9124783 5088977" 这类笔误）
                raw = m.group(1).replace(" ", "")
                try:
                    current_energy = float(raw)
                except ValueError:
                    pass

    # 最后一组
    if current_J is not None and current_alpha is not None and current_energy is not None:
        result[(current_J, current_alpha)] = current_energy

    return result


# ======================================================================
# 读取 NQS summary CSV
# ======================================================================
def read_nqs_summary(J: float, alpha_int: float) -> float | None:
    """
    在 NQS_TRAIN_DIR 中找到对应参数的 _summary.csv，返回 E_final（最低能量）。
    """
    subdir_name = f"L32_J{J}_delta0.0_alphaInt{alpha_int}"
    subdir = os.path.join(NQS_TRAIN_DIR, subdir_name)
    if not os.path.isdir(subdir):
        return None

    # 找所有 summary CSV，取最小的 E_final（对应最好的一次训练）
    energies = []
    for fname in os.listdir(subdir):
        if fname.endswith("_summary.csv"):
            fpath = os.path.join(subdir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        val = row.get("E_final", "")
                        if val and val.lower() != "nan":
                            energies.append(float(val))
            except Exception:
                pass

    return min(energies) if energies else None


# ======================================================================
# Main
# ======================================================================
def main():
    # 解析 DMRG
    dmrg = parse_dmrg(DMRG_FILE)
    if not dmrg:
        print("ERROR: No DMRG data parsed.", file=sys.stderr)
        sys.exit(1)

    # 提取 DMRG 中的唯一 J 和 alpha 列表（排序）
    j_vals = sorted(set(k[0] for k in dmrg))
    alpha_vals = sorted(set(k[1] for k in dmrg))

    print(f"DMRG points: {len(dmrg)}")
    print(f"  J values:     {j_vals}")
    print(f"  alpha values: {alpha_vals}")

    n_j = len(j_vals)
    n_a = len(alpha_vals)

    dmrg_grid = np.full((n_a, n_j), np.nan)
    nqs_grid = np.full((n_a, n_j), np.nan)

    for ai, alpha in enumerate(alpha_vals):
        for ji, J in enumerate(j_vals):
            # DMRG 能量
            if (J, alpha) in dmrg:
                dmrg_grid[ai, ji] = dmrg[(J, alpha)]

            # NQS 能量
            e_nqs = read_nqs_summary(J, alpha)
            if e_nqs is not None:
                nqs_grid[ai, ji] = e_nqs
                print(f"  NQS  J={J}, alpha={alpha}: E={e_nqs:.6f}")
            else:
                print(f"  NQS  J={J}, alpha={alpha}: MISSING")

    # 相对误差：(E_NQS - E_DMRG) / |E_DMRG|  （百分比）
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err_grid = (nqs_grid - dmrg_grid) / np.abs(dmrg_grid) * 100.0

    J_arr = np.array(j_vals)
    alpha_arr = np.array(alpha_vals)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    specs = [
        (dmrg_grid, "DMRG $E_{gs}$", "viridis", "DMRG_Energy"),
        (nqs_grid, "NQS $E_{gs}$", "viridis", "NQS_Energy"),
        (rel_err_grid, r"$(E_\mathrm{NQS}-E_\mathrm{DMRG})/|E_\mathrm{DMRG}|\,(\%)$", "RdBu_r", "RelErr"),
    ]

    for ax, (data, label, cmap, tag) in zip(axes, specs):
        # 对称色标（仅相对误差）
        if tag == "RelErr":
            vmax = np.nanmax(np.abs(data))
            im = ax.pcolormesh(J_arr, alpha_arr, data,
                               shading="nearest", cmap=cmap,
                               vmin=-vmax, vmax=vmax)
        else:
            im = ax.pcolormesh(J_arr, alpha_arr, data,
                               shading="nearest", cmap=cmap)

        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(label, fontsize=13)
        cb.ax.tick_params(labelsize=11)

        ax.set_xlabel(r"$J$", fontsize=15)
        ax.set_ylabel(r"$\alpha_\mathrm{int}$", fontsize=15)
        ax.set_xticks(J_arr)
        ax.set_yticks(alpha_arr)
        ax.tick_params(labelsize=11)

        # 单元格内标注数值
        for ai in range(n_a):
            for ji in range(n_j):
                val = data[ai, ji]
                if not np.isfinite(val):
                    continue
                if tag == "RelErr":
                    txt = f"{val:+.3f}%"
                    fontsize = 8
                else:
                    txt = f"{val:.2f}"
                    fontsize = 8
                med = np.nanmedian(data)
                color = "white" if val < med else "black"
                if tag == "RelErr":
                    color = "white" if abs(val) > np.nanmax(np.abs(data)) * 0.5 else "black"
                ax.text(J_arr[ji], alpha_arr[ai], txt,
                        ha="center", va="center",
                        fontsize=fontsize, color=color)

    axes[0].set_title(r"DMRG Ground State Energy $E_{gs}$", fontsize=13)
    axes[1].set_title(r"NQS Ground State Energy $E_{gs}$", fontsize=13)
    axes[2].set_title(r"Relative Error $\Delta E / |E_\mathrm{DMRG}|$ (%)", fontsize=13)

    plt.suptitle(r"DMRG vs NQS: $L=32$, $\delta=0$", fontsize=15, y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "svg"):
        path = os.path.join(FIG_DIR, f"compare_DMRG_NQS_L32.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    # 也单独输出一张相对误差图（更大）
    fig2, ax = plt.subplots(figsize=(8, 6))
    vmax = max(np.nanmax(np.abs(rel_err_grid)), 1e-6)
    im = ax.pcolormesh(J_arr, alpha_arr, rel_err_grid,
                       shading="nearest", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    cb = fig2.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$(E_\mathrm{NQS}-E_\mathrm{DMRG})/|E_\mathrm{DMRG}|\,(\%)$", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax.set_xlabel(r"$J$", fontsize=15)
    ax.set_ylabel(r"$\alpha_\mathrm{int}$", fontsize=15)
    ax.set_xticks(J_arr)
    ax.set_yticks(alpha_arr)
    ax.tick_params(labelsize=11)
    ax.set_title(r"NQS Relative Error vs DMRG,  $L=32$, $\delta=0$", fontsize=14)
    for ai in range(n_a):
        for ji in range(n_j):
            val = rel_err_grid[ai, ji]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(J_arr[ji], alpha_arr[ai], f"{val:+.3f}%",
                    ha="center", va="center", fontsize=9, color=color)
    plt.tight_layout()
    for ext in ("pdf", "svg"):
        path = os.path.join(FIG_DIR, f"RelErr_DMRG_NQS_L32.{ext}")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close(fig2)

    # ---- 打印汇总表 ----
    print("\n" + "=" * 75)
    print(f"{'J':>5} {'alpha':>6} {'E_DMRG':>16} {'E_NQS':>16} {'RelErr(%)':>12}")
    print("=" * 75)
    for J in j_vals:
        for alpha in alpha_vals:
            e_d = dmrg.get((J, alpha), float("nan"))
            ji = j_vals.index(J)
            ai = alpha_vals.index(alpha)
            e_n = nqs_grid[ai, ji]
            rel = rel_err_grid[ai, ji]
            print(f"{J:>5.2f} {alpha:>6.2f} {e_d:>16.6f} {e_n:>16.6f} {rel:>12.4f}")


if __name__ == "__main__":
    main()
