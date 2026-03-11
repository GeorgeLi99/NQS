#!/usr/bin/env python3
"""
计算不同参数下已训练 RBM 模型的能量涨落密度（energy fluctuation density）。

定义（参考 Trigueros et al. 2024, Simplicity of mean-field theories in neural quantum states）：
  能量涨落密度 = σ²_E / L = (⟨H²⟩ - ⟨H⟩²) / L

其中 ⟨H⟩、⟨H²⟩ 为变分态 |ψ⟩ 下的期望，通过蒙特卡洛采样估计。
哈密顿量（NQS project.md）：
  H = (Ω/2) Σ_i σ^x_i - δ Σ_i σ^z_i + Σ_{i<j} (J/r_{ij}^α) σ^z_i σ^z_j

用法:
  python3 compute_energy_fluctuation_density.py -p complex64 --L 16 --J 1.0 --delta 0.0 --alpha 2.0
  python3 compute_energy_fluctuation_density.py --list  # 扫描 train/ 下所有已有 checkpoint
"""

import argparse
import csv
import os
import sys

sys.path.append("../../")
sys.path.append("../")

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.serialization
import flax.linen as nn
import netket as nk
from netket.operator.spin import sigmax, sigmaz

_script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 默认超参数 ---
DEFAULT_PRECISION = "complex64"
DEFAULT_L = 16
DEFAULT_J = 1.0
DEFAULT_DELTA = 0.0
DEFAULT_ALPHA = 2.0
DEFAULT_ALPHA_RBM = 4
DEFAULT_CAL = 1
DEFAULT_OMEGA = 2.0

# MC 采样量（越大估计越准）
N_SAMPLES = 1024 * 32
N_CHAINS = 512
N_DISCARD = 32
CHUNK_SIZE = 1024 * 8


def _param_subdir(L: int, J: float, delta: float, alpha_int: float) -> str:
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def _basename(L: int, J: float, delta: float, alpha_int: float, alpha_rbm: int, cal: int) -> str:
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{cal}"


def build_hamiltonian(hi, L: int, J: float, delta: float, alpha_int: float, Omega: float, dtype):
    """H = (Ω/2) Σ σ^x - δ Σ σ^z + Σ_{i<j} (J/r^α) σ^z_i σ^z_j"""
    Sigma_z = np.array([[1, 0], [0, -1]], dtype=dtype)
    Sigma_x = np.array([[0, 1], [1, 0]], dtype=dtype)
    H = nk.operator.LocalOperator(hi, dtype=dtype)
    for j in range(L):
        H += (Omega / 2) * nk.operator.LocalOperator(hi, Sigma_x, [j])
    for j in range(L):
        H += (-delta) * nk.operator.LocalOperator(hi, Sigma_z, [j])
    for c in range(L):
        for j in range(c + 1, L):
            r = min(abs(c - j), L - abs(c - j))
            factor = J / r**alpha_int
            H += factor * nk.operator.LocalOperator(hi, Sigma_z, [c]) @ nk.operator.LocalOperator(hi, Sigma_z, [j])
    return H


def compute_one(
    L: int,
    J: float,
    delta: float,
    alpha_int: float,
    precision: str,
    alpha_rbm: int,
    cal: int,
    Omega: float,
) -> dict:
    """对单组参数加载 checkpoint，计算 ⟨H⟩、⟨H²⟩、能量涨落密度。"""
    dtype_np = np.complex64 if precision == "complex64" else np.complex128
    dtype_jnp = jnp.complex64 if precision == "complex64" else jnp.complex128

    g = nk.graph.Chain(length=L, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    H = build_hamiltonian(hi, L, J, delta, alpha_int, Omega, dtype_np)

    model = nk.models.RBM(
        alpha=alpha_rbm,
        param_dtype=dtype_jnp,
        use_hidden_bias=True,
        kernel_init=nn.initializers.normal(stddev=0.01),
        hidden_bias_init=nn.initializers.normal(stddev=0.01),
    )

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi,
        n_chains_per_rank=N_CHAINS,
        sweep_size=4 * L,
    )
    vs = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_discard_per_chain=N_DISCARD,
        chunk_size=CHUNK_SIZE,
        n_samples=N_SAMPLES,
    )

    param_subdir = _param_subdir(L, J, delta, alpha_int)
    base_name = _basename(L, J, delta, alpha_int, alpha_rbm, cal)
    train_dir = os.path.join(_script_dir, "train", precision, param_subdir)
    ckpt_path = os.path.join(train_dir, base_name + ".mpack")
    # 兼容旧命名：delta=0.5 时文件名可能不含 delta
    if not os.path.isfile(ckpt_path) and delta == 0.5:
        legacy_base = f"rbm_LongIsing_L={L}_J={J}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{cal}"
        alt = os.path.join(train_dir, legacy_base + ".mpack")
        if os.path.isfile(alt):
            ckpt_path = alt

    if not os.path.isfile(ckpt_path):
        return {
            "L": L, "J": J, "delta": delta, "alpha_int": alpha_int,
            "precision": precision,
            "E_mean": float("nan"), "E_var": float("nan"),
            "fluctuation_density": float("nan"),
            "error": f"checkpoint not found: {ckpt_path}",
        }

    with open(ckpt_path, "rb") as f:
        vs.variables = flax.serialization.from_bytes(vs.variables, f.read())

    # ⟨H⟩
    stat_H = vs.expect(H)
    E_mean = float(np.real(np.asarray(stat_H.mean)))

    # ⟨H²⟩ - ⟨H⟩² = 能量方差；H² 通过 H @ H 构造
    H2 = H @ H
    stat_H2 = vs.expect(H2)
    E2_mean = float(np.real(np.asarray(stat_H2.mean)))
    E_var = E2_mean - E_mean**2
    if E_var < 0:
        E_var = 0.0  # 数值误差可能导致略负

    fluctuation_density = E_var / L

    return {
        "L": L, "J": J, "delta": delta, "alpha_int": alpha_int,
        "precision": precision,
        "E_mean": E_mean, "E_var": E_var,
        "fluctuation_density": fluctuation_density,
        "error": None,
    }


def discover_checkpoints(precision: str) -> list[tuple]:
    """扫描 train/<precision>/ 下所有 .mpack，解析出 (L, J, delta, alpha_int)。"""
    import re
    train_base = os.path.join(_script_dir, "train", precision)
    if not os.path.isdir(train_base):
        return []
    found = []
    pat = re.compile(
        r"rbm_LongIsing_L=(\d+)_J=([\d.]+)_delta=([\d.]+)_alphaInt=([\d.]+)_alpha=(\d+)_Cal(\d+)\.mpack"
    )
    for subdir in os.listdir(train_base):
        d = os.path.join(train_base, subdir)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = pat.match(f)
            if m:
                L = int(m.group(1))
                J = float(m.group(2))
                delta = float(m.group(3))
                alpha_int = float(m.group(4))
                alpha_rbm = int(m.group(5))
                cal = int(m.group(6))
                found.append((L, J, delta, alpha_int, alpha_rbm, cal))
    return found


def main():
    parser = argparse.ArgumentParser(description="计算 RBM 能量涨落密度 σ²_E/L")
    parser.add_argument("-p", "--precision", default=DEFAULT_PRECISION, choices=("complex64", "complex128"))
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--J", type=float, default=DEFAULT_J)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, dest="alpha_int")
    parser.add_argument("--alpha-rbm", type=int, default=DEFAULT_ALPHA_RBM)
    parser.add_argument("--cal", type=int, default=DEFAULT_CAL)
    parser.add_argument("--Omega", type=float, default=DEFAULT_OMEGA)
    parser.add_argument("--list", action="store_true", help="扫描 train/ 下所有 checkpoint 并计算")
    parser.add_argument("-o", "--output", default=None, help="输出 CSV 路径，默认 long_range_ising/energy_fluctuation_density.csv")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    print("netket:", nk.__version__, "| backend:", jax.default_backend())

    if args.list:
        tasks = discover_checkpoints(args.precision)
        if not tasks:
            print(f"未找到 train/{args.precision}/ 下的 checkpoint")
            return
        print(f"发现 {len(tasks)} 个 checkpoint，开始计算...")
        rows = []
        for L, J, delta, alpha_int, alpha_rbm, cal in tasks:
            r = compute_one(
                L, J, delta, alpha_int, args.precision,
                alpha_rbm, cal, args.Omega,
            )
            rows.append(r)
            print(f"  L={L} J={J} δ={delta} α={alpha_int}: E={r['E_mean']:.6g} σ²_E/L={r['fluctuation_density']:.6e}")
    else:
        r = compute_one(
            args.L, args.J, args.delta, args.alpha_int, args.precision,
            args.alpha_rbm, args.cal, args.Omega,
        )
        rows = [r]
        if r.get("error"):
            print("错误:", r["error"], file=sys.stderr)
        else:
            print(f"E_mean = {r['E_mean']:.8g}")
            print(f"E_var  = {r['E_var']:.8g}")
            print(f"能量涨落密度 σ²_E/L = {r['fluctuation_density']:.8e}")

    out_path = args.output or os.path.join(_script_dir, "energy_fluctuation_density.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["L", "J", "delta", "alpha_int", "precision", "E_mean", "E_var", "fluctuation_density", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
