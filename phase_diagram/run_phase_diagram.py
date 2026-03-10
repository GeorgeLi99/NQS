#!/usr/bin/env python3
"""
批量训练 Long-range Ising RBM：蛇形遍历 (alphaInt, J) 网格 + 迁移学习。

逻辑照搬 long_range_ising/rbm_long_range_ising.py，但在一次进程里依次
训练所有 (J, alphaInt) 组合，每个点从上一个相邻点的 checkpoint 初始化。

所有输出放在 phase_diagram/train/<PRECISION>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/
"""

import sys
sys.path.append("../../")
sys.path.append("../")

import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.serialization
import flax.linen as nn
import optax as opx
import time

from netket.operator.spin import sigmax, sigmaz

print("netket version:", nk.__version__)
print(f"Using platform: {jax.default_backend()}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALPHA_INT_LIST, J_LIST,
    L, delta, Omega, PRECISION, alpha_rbm, key_cal, use_bias,
    N_ITER_FIRST, N_ITER_TRANSFER,
    val_learning_rate, val_diagonal_shift,
    N_samples, n_chains_per_rank, n_discard_per_chain, chunk_size,
    param_subdir as _param_subdir, file_base as _file_base,
)

dtype_np = np.complex64 if PRECISION == "complex64" else np.complex128
dtype_jnp = jnp.complex64 if PRECISION == "complex64" else jnp.complex128

_script_dir = os.path.dirname(os.path.abspath(__file__))


def _train_dir(J: float, alpha_int: float) -> str:
    d = os.path.join(_script_dir, "train", PRECISION, _param_subdir(J, alpha_int))
    os.makedirs(d, exist_ok=True)
    return d


def _mpack_path(J: float, alpha_int: float) -> str:
    return os.path.join(_train_dir(J, alpha_int), _file_base(J, alpha_int) + ".mpack")


def _log_base(J: float, alpha_int: float) -> str:
    return os.path.join(_train_dir(J, alpha_int), _file_base(J, alpha_int))


# ======================================================================
# 构建蛇形遍历顺序
# ======================================================================
def build_snake_order(alpha_list: list, j_list: list) -> list[tuple[float, float]]:
    """返回 [(J, alphaInt), ...] 的蛇形遍历列表。"""
    order = []
    for idx, alpha in enumerate(alpha_list):
        js = j_list if idx % 2 == 0 else list(reversed(j_list))
        for j in js:
            order.append((j, alpha))
    return order


# ======================================================================
# 构建 Hamiltonian & observables（每换参数需重建）
# ======================================================================
def build_hamiltonian(hi, J_val: float, alpha_int: float):
    Sigma_z = np.array([[1, 0], [0, -1]], dtype=dtype_np)
    Sigma_x = np.array([[0, 1], [1, 0]], dtype=dtype_np)

    H = nk.operator.LocalOperator(hi, dtype=dtype_np)
    for j in range(L):
        H += (Omega / 2) * nk.operator.LocalOperator(hi, Sigma_x, [j])
    for j in range(L):
        H += (-delta) * nk.operator.LocalOperator(hi, Sigma_z, [j])
    for c in range(L):
        for j in range(c + 1, L):
            r = min(abs(c - j), L - abs(c - j))
            factor = J_val / r ** alpha_int
            H += factor * nk.operator.LocalOperator(hi, Sigma_z, [c]) @ nk.operator.LocalOperator(hi, Sigma_z, [j])
    return H


def build_observables(hi):
    Mx = nk.operator.LocalOperator(hi, dtype=dtype_np)
    Mz = nk.operator.LocalOperator(hi, dtype=dtype_np)
    Mz_AFM = nk.operator.LocalOperator(hi, dtype=dtype_np)
    for j in range(L):
        Mx += (1 / L) * sigmax(hi, j)
        Mz += (1 / L) * sigmaz(hi, j)
        Mz_AFM += ((1 / L) * (-1) ** j) * sigmaz(hi, j)
    return {"Mx": Mx, "Mz": Mz, "Mz_AFM": Mz_AFM}


# ======================================================================
# Main training loop
# ======================================================================
def main():
    total_start = time.time()

    g = nk.graph.Chain(length=L, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    model = nk.models.RBM(
        alpha=alpha_rbm,
        param_dtype=dtype_jnp,
        use_hidden_bias=True,
        kernel_init=nn.initializers.normal(stddev=0.01),
        hidden_bias_init=nn.initializers.normal(stddev=0.01),
    )

    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=n_chains_per_rank, sweep_size=4 * L)
    opt = nk.optimizer.Sgd(learning_rate=val_learning_rate)
    sr = nk.optimizer.SR(diag_shift=val_diagonal_shift, holomorphic=True)

    order = build_snake_order(ALPHA_INT_LIST, J_LIST)
    total = len(order)
    print(f"\n{'=' * 70}")
    print(f"Phase diagram: {total} sampling points (snake order)")
    print(f"  ALPHA_INT_LIST = {ALPHA_INT_LIST}")
    print(f"  J_LIST         = {J_LIST}")
    print(f"  L={L}, delta={delta}, Omega={Omega}, PRECISION={PRECISION}")
    print(f"  N_ITER first={N_ITER_FIRST}, transfer={N_ITER_TRANSFER}")
    print(f"{'=' * 70}\n")

    prev_mpack: str | None = None

    for idx, (J_val, alpha_int) in enumerate(order):
        point_start = time.time()
        n_iter = N_ITER_FIRST if idx == 0 else N_ITER_TRANSFER
        print(f"\n[{idx + 1}/{total}]  J={J_val}, alphaInt={alpha_int}  ({n_iter} iters)")

        H = build_hamiltonian(hi, J_val, alpha_int)
        obs = build_observables(hi)

        vs = nk.vqs.MCState(
            sampler=sampler, model=model,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size, n_samples=N_samples,
        )

        # Transfer learning: load previous checkpoint
        if prev_mpack is not None and os.path.isfile(prev_mpack):
            with open(prev_mpack, "rb") as f:
                vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
            print(f"  Loaded checkpoint: {prev_mpack}")
        elif prev_mpack is not None:
            print(f"  WARNING: previous checkpoint not found: {prev_mpack}, training from scratch")

        vmc = nk.VMC(hamiltonian=H, optimizer=opt, variational_state=vs, preconditioner=sr)

        out_base = _log_base(J_val, alpha_int)
        mpack_out = _mpack_path(J_val, alpha_int)
        print(f"  Output: {out_base}.log")

        vmc.run(out=out_base, n_iter=n_iter, obs=obs)

        prev_mpack = mpack_out
        elapsed = time.time() - point_start
        print(f"  Done in {elapsed:.1f}s.  Checkpoint: {mpack_out}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All {total} points finished in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
