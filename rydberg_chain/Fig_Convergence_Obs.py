import argparse
import csv
import os
import sys
import numpy as np
import scienceplots

# 数据路径：train/<precision>/L{L}_Rb{Rb}_delta{delta}_alpha{alpha}/ 下 merge 后的 CSV
_script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 默认超参数（可由命令行覆盖）---
DEFAULT_PRECISION = "complex64"
DEFAULT_L = 16
DEFAULT_Rb = 1.0
DEFAULT_DELTA = 0.5
DEFAULT_ALPHA = 6.0

parser = argparse.ArgumentParser(description="绘制 Rydberg VMC 收敛与观测量")
parser.add_argument("--precision", "-p", choices=("complex64", "complex128"), default=DEFAULT_PRECISION,
                    help=f"精度子目录（默认: {DEFAULT_PRECISION}）")
parser.add_argument("--L", type=int, default=DEFAULT_L, help=f"链长（默认: {DEFAULT_L}）")
parser.add_argument("--Rb", type=float, default=DEFAULT_Rb, help=f"Blockade radius（默认: {DEFAULT_Rb}）")
parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help=f"Detuning（默认: {DEFAULT_DELTA}）")
parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"相互作用指数 α（默认: {DEFAULT_ALPHA}）")
args, _ = parser.parse_known_args()
PRECISION = args.precision
_param_subdir = f"L{args.L}_Rb{args.Rb}_delta{args.delta}_alpha{args.alpha}"
BASENAME = f"rydberg_L{args.L}_delta{args.delta}_Rb{args.Rb}_alpha{args.alpha}"
DIR_DATA = os.path.join(_script_dir, "train", PRECISION, _param_subdir)


def _resolve_parsed_csv():
    for name in [f"{BASENAME}_merged_parsed.csv", f"{BASENAME}_run1_parsed.csv", f"{BASENAME}_parsed.csv"]:
        p = os.path.join(DIR_DATA, name)
        if os.path.isfile(p):
            return p
    return os.path.join(DIR_DATA, f"{BASENAME}_merged_parsed.csv")


def _resolve_summary_csv():
    for name in [f"{BASENAME}_merged_summary.csv", f"{BASENAME}_run1_summary.csv", f"{BASENAME}_summary.csv"]:
        p = os.path.join(DIR_DATA, name)
        if os.path.isfile(p):
            return p
    return os.path.join(DIR_DATA, f"{BASENAME}_merged_summary.csv")


PARSED_CSV = _resolve_parsed_csv()
SUMMARY_CSV = _resolve_summary_csv()


def _read_parsed_csv(parsed_path: str):
    """从 _parsed.csv 读成列字典。"""
    with open(parsed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {parsed_path}")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


def load_convergence_from_csv(parsed_path: str, E0: float):
    """从 parsed/merged CSV 读取迭代轴、Energy，计算相对误差 ε = |E-E0|/|E0|。优先用 global_iter。"""
    cols = _read_parsed_csv(parsed_path)
    iters = cols["global_iter"] if "global_iter" in cols else cols["iter"]
    energy = cols["Energy"]
    relative_error = np.abs((energy - E0) / E0)
    return iters, energy, relative_error


def load_observables_from_csv(parsed_path: str):
    """从 parsed/merged CSV 读取迭代轴、Ntot、Mz。优先用 global_iter。"""
    cols = _read_parsed_csv(parsed_path)
    iters = cols["global_iter"] if "global_iter" in cols else cols["iter"]
    return iters, cols["Ntot"], cols["Mz"]


def load_E0_from_summary(summary_path: str):
    """从 summary/merged CSV 读取参考能量。合并文件多行时取最后一行 E_final。"""
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return float(rows[-1]["E_final"])


### main code
L = args.L
alpha = args.alpha
delta = args.delta

# 参考基态能量：优先从 summary CSV 读 E_final，否则用已知值
E0 = load_E0_from_summary(SUMMARY_CSV)
if E0 is None:
    E0 = -8.878144715543531

plot_key = "Energy_and_Obs"

print("Precision (data dir):", PRECISION)
print("Data:", PARSED_CSV)
print("GS energy (E0):", E0)

# 从 CSV 读取收敛与观测量
iters, energy, relative_error = load_convergence_from_csv(PARSED_CSV, E0)
data1 = (iters, relative_error)

iters_obs, ntot, mz = load_observables_from_csv(PARSED_CSV)
dataObs1 = (iters_obs, ntot, mz)


colors = ["#085293", "#90d4bd",
    "#f58b47", "#fcce25",
    "#6300a7", "#a51f99",
    "#b7ea63",]

# plot the figure（Science 风格，需安装 SciencePlots: pip install SciencePlots）
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])  # 不用 LaTeX，避免未安装时报错

# figure size
plt.figure(figsize=(2 * 8.6, 6.45))

# (1) relative error
ax1=plt.subplot(121)

ax1.plot(data1[0], data1[1], color=colors[0], lw=2.0, 
         label=rf"$\delta={delta}, \alpha = $"+f" {alpha}",)

ax1.set_ylabel(r'$\epsilon = |\frac{E-E_0}{E_0}|$',
               fontsize=25)
ax1.set_xlabel('Iteration',fontsize=25)

ax1.set_xlim((0,2000))
ax1.set_ylim((1e-8, 1))
ax1.set_yscale('log')

ax1.set_xticks([0, 500, 1000, 1500, 2000],
    ['0',r'500',r'1000',r'1500', r'2000'],
    fontsize=25,)

ax1.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$', r'$10^{-6}$',  r'$10^{-7}$'],
    fontsize=25,)


ax1.legend(loc='upper right',fontsize=25, frameon=False,)
ax1.text(0.02, 0.02, rf"$E_0 = {E0:.6g}$", transform=ax1.transAxes, fontsize=18, verticalalignment="bottom")

ax1.tick_params("both", which='major', length=4, # width=1.0, 
    direction='in',#labelsize=12.5
    )
ax1.tick_params("both", which='minor', length=2, # width=1.0, 
    direction='in',#labelsize=12.5
    )

# (2) observables
ax2=plt.subplot(122)

ax2.plot(data1[0], np.abs(dataObs1[1]), color=colors[3], lw=2.0, 
         label=r"$D=0.0, |N_{tot}|, \alpha=$"+f" {alpha}", )
ax2.plot(data1[0], np.abs(dataObs1[2]), color=colors[4], lw=2.0, 
         label=r"$D=0.0, |M_z|, \alpha=$"+f" {alpha}", )

ax2.set_ylabel(r'$M_z = \frac{1}{L}\sum_{j=1}^L \langle \sigma^z_j\rangle$',
               fontsize=25)
ax2.set_xlabel('Iteration',fontsize=25)

ax2.set_xlim((0,2000))
ax2.set_ylim((1e-5,1.0))

# ax1.set_yscale('log')

ax2.set_xticks([0, 500, 1000, 1500, 2000],
    ['0',r'500',r'1000',r'1500', r'2000'],
    fontsize=25,)

ax2.legend(loc='upper right',fontsize=25, frameon=False,)

ax2.tick_params("both", which='major', length=4, # width=1.0, 
    direction='in',#labelsize=12.5
    )
ax2.tick_params("both", which='minor', length=2, # width=1.0, 
    direction='in',#labelsize=12.5
    )

# Big title
plt.suptitle(r"$\delta = $"+f"{delta}, "+f"$L=${L}, RBM, "+r"$\alpha=$"+f"{alpha}", fontsize=25)
# adjust the space
plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.tight_layout()

# 先保存再显示，避免 show() 后图形被清空；输出到 rydberg_chain/figure/
out_dir = os.path.join(_script_dir, "figure")
os.makedirs(out_dir, exist_ok=True)
basename_pdf = f"Fig_ConvObs_RBM_delta={delta}_L={L}_{plot_key}.pdf"
basename_svg = f"Fig_ConvObs_RBM_delta={delta}_L={L}_{plot_key}.svg"


def _save_fig(path: str) -> bool:
    """保存到 path，失败返回 False（如权限或文件被占用）。"""
    try:
        plt.savefig(path)
        return True
    except (PermissionError, OSError) as e:
        print(f"  保存失败 {path}: {e}", file=sys.stderr)
        return False


saved_any = False
for ext, basename in [("pdf", basename_pdf), ("svg", basename_svg)]:
    path = os.path.join(out_dir, basename)
    if _save_fig(path):
        print(f"  已保存: {path}")
        saved_any = True
    else:
        # 权限不足或文件被占用时，尝试保存到脚本所在目录
        fallback = os.path.join(_script_dir, basename)
        if _save_fig(fallback):
            print(f"  已保存到备用路径: {fallback}")
            saved_any = True
if not saved_any:
    print("  警告: 未能保存图片，请关闭已打开的 PDF/SVG 或检查目录权限。")
plt.show()