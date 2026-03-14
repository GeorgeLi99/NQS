import sys
import os
sys.path.append('../../')
sys.path.append('../')
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import json
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import flax.serialization  # noqa: F401 - 用于 from_bytes，部分类型桩未导出
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from mpi4py import MPI
from netket.operator.spin import sigmax, sigmaz
from netket.utils.group import PermutationGroup, Permutation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("netket version:", nk.__version__)
print(f"Using platform: {jax.default_backend()}")
start_time = time.time()

# ======================================================================
# 超参数（集中放在文件最前，便于修改）
# ======================================================================
# 物理 / 模型
J = 2.0
alpha_interaction = 0.5
L = 32
alpha_rbm = 12
use_bias = False
key_cal = 1

# 采样与 VQS
N_samples = 1024 * 64
n_chains_per_rank = 512
n_discard_per_chain = 32
chunk_size = 1024 * 8

# 关联函数：起点与半链长（site_end = site_start + L_half）
site_start = 7
L_half = L // 2
site_end = site_start + L_half

# Checkpoint 文件名（或改为路径；若用 config 则见下方注释）
load_name = "rbm_LongIsing_L=32_J=2.0_delta=0.0_alphaInt=0.5_alpha=12_Cal1.mpack"
# 若从 phase_diagram/config 与 train 目录约定生成路径，可改为：
# from config import param_subdir, file_base, PRECISION, OUTPUT_ROOT
# _root = os.environ.get("PHASE_DIAGRAM_OUTPUT_ROOT") or OUTPUT_ROOT or os.path.dirname(os.path.abspath(__file__))
# load_name = os.path.join(_root, "train", PRECISION, param_subdir(J, alpha_interaction), file_base(J, alpha_interaction) + ".mpack")

# ======================================================================
# 以下为计算与模型构建
# ======================================================================

def n_site_translation(L, n):
    """
    L: length of spin chain (even)
    n: translation number
    """
    return [(i + n) % L for i in range(L)]

def spatial_inversion(L):
    """ Inversion operation. """
    identity = [i for i in range(L)]
    print("Identity element:", identity)
    inv = identity[::-1]
    return inv

from netket.utils.group import PermutationGroup, Permutation
# Representation is not supported in Netket3.13
# from netket.symmetry import Representation

def OneSiteTranslationGroup(L: int):
    """Wrapping into netket permutation group."""
    Nc = L
    group_elems_T1 = []
    for k in range(Nc):
        permu_array = n_site_translation(L, k)
        group_elems_T1.append(Permutation(np.array(permu_array, dtype=np.int32), name=f"T({k})"))
    return PermutationGroup(elems=group_elems_T1, degree=L)


print(f"Ising J = {J}, alpha_interaction = {alpha_interaction}, L = {L}")
print(f"alpha_rbm = {alpha_rbm}, use_bias = {use_bias}, key_cal = {key_cal}")

# Graph
g = nk.graph.Chain(length=L, pbc=True)
# print("Translation group:\n ",g.translation_group())

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5,  N=g.n_nodes, #total_sz= 0, 
    ) 

# Hamiltonian
Sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

H = nk.operator.LocalOperator(hi, dtype=complex)
# (1) trasverse field 
for j in range(L):
    H += nk.operator.LocalOperator(hi, Sigma_x, [j]) 
# (2) long-range Ising interaction
for c in range(L):
    # print(f"site_c: c = {c}")
    # summation for i < j
    for j in range(c+1, L):  
        #print(f"site_j: j = {j}")
        r = min(abs(c-j), L - abs(c-j))
        #print(f"Distance R = {r}")
        factor = J/r**alpha_interaction
        H += factor * nk.operator.LocalOperator(hi, Sigma_z, [c]) @ nk.operator.LocalOperator(hi, Sigma_z, [j])

# 平移对称群（使用文件顶部超参 L）
translation_group = OneSiteTranslationGroup(L)
print("Translation group: L =", L, "elements")

model = nk.models.RBMSymm(
    alpha=alpha_rbm,
    param_dtype=jnp.complex64,
    use_hidden_bias=True,
    use_visible_bias=use_bias,
    symmetries=translation_group,
    kernel_init=nn.initializers.normal(stddev=0.01),
    hidden_bias_init=nn.initializers.normal(stddev=0.01),
)
print("RBM with symmetry: alpha =", alpha_rbm)

# MC 采样器与 VQS（N_samples, n_chains_per_rank 等见文件顶部超参）
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=n_chains_per_rank, sweep_size=4 * L)
print("Sampler:", sampler)

vs = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size,
    n_samples=N_samples,
)
print("Variational state:", vs)
print("Number of parameters:", vs.n_parameters)

# 加载 checkpoint（load_name 见文件顶部超参）
with open(load_name, "rb") as file:
    vs.variables = flax.serialization.from_bytes(vs.variables, file.read())

##############################################################################
# observable: Correlation function  <σ^z_{site_start} σ^z_k>,  k = site_start .. site_end-1
corr_func_info = {}
corr_func_val = np.zeros(L_half, dtype=np.complex64)
for k in range(site_start, site_end):
    corr_op = nk.operator.LocalOperator(hi, Sigma_z, [site_start]) @ nk.operator.LocalOperator(hi, Sigma_z, [k])
    corr_k = vs.expect(corr_op)
    corr_func_val[k - site_start] = corr_k.mean
    corr_func_info[f"Czz_{site_start}_{k}"] = corr_k
print(f"Corr func (Mean values): {corr_func_val}")

# save
np.savetxt('Czz_mean.txt', corr_func_val, fmt='%.8f', delimiter='\t', header='zz-spin correlation function')
with open('Czz_info.json', 'w') as f:
    json.dump(corr_func_info, f)

elapsed = time.time() - start_time
print(f"\nCorrelation function computed in {elapsed:.1f}s")

##############################################################################
# 绘图：C_zz(j) vs j
##############################################################################
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    print("matplotlib/scienceplots 不可用，跳过绘图。")
    sys.exit(0)

base_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(base_dir, "figure")
os.makedirs(fig_dir, exist_ok=True)

plt.style.use(["science", "no-latex"])

j_arr = np.arange(L_half)
corr_real = np.real(corr_func_val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：C_zz(j) 原始值
ax1.plot(j_arr, corr_real, "o-", markersize=4, color="#2c7bb6")
ax1.axhline(y=0, color="gray", ls="--", lw=0.8)
ax1.set_xlabel(r"$j - j_0$", fontsize=16)
ax1.set_ylabel(rf"$\langle \sigma^z_{{{site_start}}} \sigma^z_{{j}} \rangle$", fontsize=16)
ax1.set_title(rf"$C_{{zz}}(j)$,  $J={J}$, $\alpha={alpha_interaction}$, $L={L}$", fontsize=14)
ax1.tick_params(labelsize=12)

# 右图：|C_zz(j)|, 对数纵轴，看衰减行为
abs_corr = np.abs(corr_real)
ax2.semilogy(j_arr[1:], abs_corr[1:], "s-", markersize=4, color="#d7191c")
ax2.set_xlabel(r"$j - j_0$", fontsize=16)
ax2.set_ylabel(r"$|C_{zz}(j)|$", fontsize=16)
ax2.set_title(rf"$|C_{{zz}}(j)|$  (log scale)", fontsize=14)
ax2.tick_params(labelsize=12)

plt.tight_layout()

fig_base = f"Czz_J{J}_alpha{alpha_interaction}_L{L}"
for ext in ("pdf", "svg"):
    path = os.path.join(fig_dir, f"{fig_base}.{ext}")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
plt.close(fig)

# ---- (-1)^j C_zz(j)：凸显 AFM 长程序 ----
fig2, ax = plt.subplots(figsize=(8, 5))
staggered = np.array([(-1) ** dj * corr_real[dj] for dj in range(L_half)])
ax.plot(j_arr, staggered, "D-", markersize=4, color="#1a9641")
ax.axhline(y=0, color="gray", ls="--", lw=0.8)
ax.set_xlabel(r"$j - j_0$", fontsize=16)
ax.set_ylabel(rf"$(-1)^{{j-j_0}} \langle \sigma^z_{{{site_start}}} \sigma^z_j \rangle$", fontsize=16)
ax.set_title(
    rf"Staggered $C_{{zz}}$,  $J={J}$, $\alpha={alpha_interaction}$, $L={L}$",
    fontsize=13,
)
ax.tick_params(labelsize=12)
plt.tight_layout()

for ext in ("pdf", "svg"):
    path = os.path.join(fig_dir, f"{fig_base}_staggered.{ext}")
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
plt.close(fig2)

print("\nDone.")