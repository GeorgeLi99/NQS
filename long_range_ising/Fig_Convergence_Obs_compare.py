"""
将 complex128 与 complex64 的 Long-range Ising VMC 结果（合并后的 CSV）绘于同一张图。
左图：相对误差 ε = |E-E0|/|E0| vs 迭代（complex128, complex64）
右图：|Mx|、|Mz| vs 迭代（各两条：complex128 与 complex64）
数据路径：long_range_ising/train/<precision>/ 下 _merged_parsed.csv
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

_script_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(_script_dir, "train")
BASENAME = "rbm_LongIsing_L=16_J=1.0_alphaInt=2.0_alpha=4_Cal1"


def _resolve_parsed_csv(dir_path: str):
    for name in [f"{BASENAME}_merged_parsed.csv", f"{BASENAME}_run1_parsed.csv", f"{BASENAME}_parsed.csv"]:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p
    return os.path.join(dir_path, f"{BASENAME}_merged_parsed.csv")


def _resolve_summary_csv(dir_path: str):
    for name in [f"{BASENAME}_merged_summary.csv", f"{BASENAME}_run1_summary.csv", f"{BASENAME}_summary.csv"]:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p
    return os.path.join(dir_path, f"{BASENAME}_merged_summary.csv")


def _read_parsed_csv(parsed_path: str):
    with open(parsed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {parsed_path}")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def load_convergence_from_csv(parsed_path: str, E0: float):
    cols = _read_parsed_csv(parsed_path)
    iters = cols["global_iter"] if "global_iter" in cols else cols["iter"]
    energy = cols["Energy"]
    relative_error = np.abs((energy - E0) / E0)
    return iters, energy, relative_error


def load_observables_from_csv(parsed_path: str):
    cols = _read_parsed_csv(parsed_path)
    iters = cols["global_iter"] if "global_iter" in cols else cols["iter"]
    mx = cols.get("Mx")
    mz = cols.get("Mz")
    if mx is None or mz is None:
        raise ValueError("CSV 需包含 Mx, Mz 列")
    return iters, mx, mz


def load_E0_from_summary(summary_path: str):
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return float(rows[-1]["E_final"])


L = 16
alpha_int = 2.0
plot_key = "Energy_and_Obs_compare"

dir_128 = os.path.join(TRAIN_DIR, "complex128")
dir_64 = os.path.join(TRAIN_DIR, "complex64")
parsed_128 = _resolve_parsed_csv(dir_128)
parsed_64 = _resolve_parsed_csv(dir_64)
summary_128 = _resolve_summary_csv(dir_128)

E0 = load_E0_from_summary(summary_128)
if E0 is None:
    E0 = -5.0

print("E0:", E0)
print("complex128 (merged):", parsed_128)
print("complex64 (merged):", parsed_64)

iters_128, _, rel_err_128 = load_convergence_from_csv(parsed_128, E0)
_, mx_128, mz_128 = load_observables_from_csv(parsed_128)

iters_64, _, rel_err_64 = load_convergence_from_csv(parsed_64, E0)
_, mx_64, mz_64 = load_observables_from_csv(parsed_64)

colors = ["#085293", "#90d4bd", "#f58b47", "#fcce25", "#6300a7", "#a51f99", "#b7ea63"]

plt.style.use(["science", "no-latex"])
plt.figure(figsize=(2 * 8.6, 6.45))

ax1 = plt.subplot(121)
ax1.plot(iters_128, rel_err_128, color=colors[0], lw=2.0, label="complex128")
ax1.plot(iters_64, rel_err_64, color=colors[1], lw=2.0, label="complex64")
ax1.set_ylabel(r'$\epsilon = |\frac{E-E_0}{E_0}|$', fontsize=25)
ax1.set_xlabel("Iteration", fontsize=25)
ax1.set_xlim((0, 2000))
ax1.set_ylim((1e-8, 1))
ax1.set_yscale("log")
ax1.set_xticks([0, 500, 1000, 1500, 2000],
               ["0", r"500", r"1000", r"1500 ", r"2000"], fontsize=25)
ax1.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
               [r"$10^0$", r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$", r"$10^{-5}$", r"$10^{-6}$", r"$10^{-7}$"],
               fontsize=25)
ax1.legend(loc="upper right", fontsize=25, frameon=False)
ax1.text(0.02, 0.02, rf"$E_0 = {E0:.6g}$", transform=ax1.transAxes, fontsize=18, verticalalignment="bottom")
ax1.tick_params("both", which="major", length=4, direction="in")
ax1.tick_params("both", which="minor", length=2, direction="in")

ax2 = plt.subplot(122)
ax2.plot(iters_128, np.abs(mx_128), color=colors[2], lw=2.0, label=r"complex128 $|M_x|$")
ax2.plot(iters_128, np.abs(mz_128), color=colors[3], lw=2.0, label=r"complex128 $|M_z|$")
ax2.plot(iters_64, np.abs(mx_64), color=colors[4], lw=2.0, linestyle="--", label=r"complex64 $|M_x|$")
ax2.plot(iters_64, np.abs(mz_64), color=colors[5], lw=2.0, linestyle="--", label=r"complex64 $|M_z|$")
ax2.set_ylabel(r'$M_x,\, |M_z|$', fontsize=25)
ax2.set_xlabel("Iteration", fontsize=25)
ax2.set_xlim((0, 2000))
ax2.set_ylim((1e-5, 1.0))
ax2.set_xticks([0, 500, 1000, 1500, 2000], ["0", "500", "1000", "1500", "2000"], fontsize=25)
ax2.legend(loc="upper right", fontsize=20, frameon=False)
ax2.tick_params("both", which="major", length=4, direction="in")
ax2.tick_params("both", which="minor", length=2, direction="in")

plt.suptitle(r"Long-range Ising: $\delta = 0.5,\, L=$" + f"{L}, RBM, " + r"$\alpha_{int}=$" + f"{alpha_int} (complex128 vs complex64)", fontsize=25)
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.tight_layout()

out_dir = os.path.join(_script_dir, "figure")
os.makedirs(out_dir, exist_ok=True)
basename_pdf = f"Fig_ConvObs_RBM_LongIsing_delta=0.5_L={L}_{plot_key}.pdf"
basename_svg = f"Fig_ConvObs_RBM_LongIsing_delta=0.5_L={L}_{plot_key}.svg"


def _save_fig(path: str) -> bool:
    try:
        plt.savefig(path)
        return True
    except (PermissionError, OSError) as e:
        print(f"  保存失败 {path}: {e}", file=sys.stderr)
        return False


saved_any = False
for basename in [basename_pdf, basename_svg]:
    path = os.path.join(out_dir, basename)
    if _save_fig(path):
        print(f"  已保存: {path}")
        saved_any = True
    else:
        fallback = os.path.join(_script_dir, basename)
        if _save_fig(fallback):
            print(f"  已保存到备用路径: {fallback}")
            saved_any = True
if not saved_any:
    print("  警告: 未能保存图片，请关闭已打开的 PDF/SVG 或检查目录权限。")
plt.show()
