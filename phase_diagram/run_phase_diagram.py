#!/usr/bin/env python3
"""
批量训练 Long-range Ising RBM：蛇形遍历 (alphaInt, J) 网格 + 迁移学习。

逻辑照搬 long_range_ising/rbm_long_range_ising.py，但在一次进程里依次
训练所有 (J, alphaInt) 组合，每个点从上一个相邻点的 checkpoint 初始化。

所有输出放在 phase_diagram/train/<PRECISION>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/
"""

from __future__ import annotations

import sys
import os
import time
import subprocess
from typing import Any, cast

sys.path.append("../../")
sys.path.append("../")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALPHA_INT_LIST, J_LIST,
    L, delta, Omega, PRECISION, alpha_rbm, key_cal, use_bias,
    N_ITER_FIRST, N_ITER_TRANSFER,
    val_learning_rate, val_diagonal_shift,
    N_samples, n_chains_per_rank, n_discard_per_chain, chunk_size,
    param_subdir as _param_subdir, file_base as _file_base,
    EARLY_STOP_WINDOW, EARLY_STOP_TOL,
    USE_LR_SCHEDULE, LR_DECAY_ALPHA,
    USE_TWO_GPUS,
    OUTPUT_ROOT,
)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_output_root = os.environ.get("PHASE_DIAGRAM_OUTPUT_ROOT") or OUTPUT_ROOT or _script_dir
_output_root = os.path.abspath(os.path.expanduser(str(_output_root)))

# ----------------------------------------------------------------------
# 两张 GPU 并行模式：在导入 JAX 之前就启动子进程并退出父进程
# ----------------------------------------------------------------------
def _split_alpha_for_two_gpus(alpha_list: list[float]) -> tuple[list[float], list[float]]:
    a0: list[float] = []
    a1: list[float] = []
    for i, a in enumerate(alpha_list):
        (a0 if i % 2 == 0 else a1).append(a)
    return a0, a1


def _spawn_worker(gpu_id: int, alpha_subset: list[float]) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("JAX_PLATFORM_NAME", "gpu")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", ",".join(str(x) for x in alpha_subset)]
    return subprocess.Popen(cmd, env=env)


if __name__ == "__main__" and USE_TWO_GPUS and (len(sys.argv) < 2 or sys.argv[1] != "--worker"):
    a0, a1 = _split_alpha_for_two_gpus([float(x) for x in ALPHA_INT_LIST])
    print("Two-GPU enabled. alphaInt split:")
    print("  GPU0:", a0)
    print("  GPU1:", a1)
    p0 = _spawn_worker(0, a0)
    p1 = _spawn_worker(1, a1)
    rc0 = p0.wait()
    rc1 = p1.wait()
    raise SystemExit(0 if (rc0 == 0 and rc1 == 0) else f"Two-GPU run failed: rc0={rc0}, rc1={rc1}")

# ----------------------------------------------------------------------
# 下面才导入 JAX / NetKet（保证每个进程已绑定好 GPU）
# ----------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.serialization
import flax.linen as nn
import optax as opx

from netket.operator.spin import sigmax, sigmaz

print("netket version:", nk.__version__)
print(f"Using platform: {jax.default_backend()}")

dtype_np = np.complex64 if PRECISION == "complex64" else np.complex128
dtype_jnp = jnp.complex64 if PRECISION == "complex64" else jnp.complex128

def make_optimizer(n_iter: int) -> Any:
    if not USE_LR_SCHEDULE:
        return nk.optimizer.Sgd(learning_rate=float(val_learning_rate))

    lr_schedule = opx.cosine_decay_schedule(
        init_value=float(val_learning_rate),
        decay_steps=int(n_iter),
        alpha=float(LR_DECAY_ALPHA),
    )
    # NetKet 的类型标注通常把 learning_rate 写死为 float，但运行时支持 schedule callable
    return nk.optimizer.Sgd(learning_rate=cast(Any, lr_schedule))


def _energy_mean(driver: nk.VMC) -> float:
    """
    返回当前 step 的能量均值（尽量稳健地转成 python float）。
    """
    e: Any = driver.energy.mean
    # 可能是复数/DeviceArray；我们只取实部
    return float(np.real(np.asarray(e)))


def _train_dir(J: float, alpha_int: float) -> str:
    d = os.path.join(_output_root, "train", PRECISION, _param_subdir(J, alpha_int))
    try:
        os.makedirs(d, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"{e}\n"
            f"当前输出目录不可写：{d}\n"
            f"请在服务器上设置可写输出根目录：\n"
            f"  方式1（推荐）：export PHASE_DIAGRAM_OUTPUT_ROOT=/path/to/writable/phase_diagram\n"
            f"  方式2：在 phase_diagram/config.py 设置 OUTPUT_ROOT = \"/path/to/writable/phase_diagram\"\n"
        ) from e
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
def main(alpha_subset: list[float] | None = None):
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
    sr = nk.optimizer.SR(diag_shift=val_diagonal_shift, holomorphic=True)

    alpha_list = ALPHA_INT_LIST if alpha_subset is None else alpha_subset
    order = build_snake_order(alpha_list, J_LIST)
    total = len(order)
    print(f"\n{'=' * 70}")
    print(f"Phase diagram: {total} sampling points (snake order)")
    print(f"  ALPHA_INT_LIST = {alpha_list}")
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

        opt = make_optimizer(n_iter)
        vmc = nk.VMC(hamiltonian=H, optimizer=opt, variational_state=vs, preconditioner=sr)

        out_base = _log_base(J_val, alpha_int)
        mpack_out = _mpack_path(J_val, alpha_int)
        print(f"  Output: {out_base}.log")
        if USE_LR_SCHEDULE:
            print(f"  LR schedule: cosine decay, lr0={val_learning_rate}, alpha_end={LR_DECAY_ALPHA}, max_steps={n_iter}")
        else:
            print(f"  LR schedule: constant lr={val_learning_rate}")
        print(f"  Early stop: window={EARLY_STOP_WINDOW}, tol={EARLY_STOP_TOL}")

        energies: list[float] = []
        stopped_early = False

        # 逐回合训练，以实现“最近 100 回合能量改善不足阈值则早停”
        for it in range(1, n_iter + 1):
            vmc.run(out=out_base, n_iter=1, obs=obs)
            e_now = _energy_mean(vmc)
            energies.append(e_now)

            if len(energies) > EARLY_STOP_WINDOW:
                prev_best = float(np.min(energies[-(EARLY_STOP_WINDOW + 1):-1]))
                improve = prev_best - e_now
                if improve < EARLY_STOP_TOL:
                    print(
                        f"  Early stopping at iter {it}/{n_iter}: "
                        f"best(prev {EARLY_STOP_WINDOW})={prev_best:.12g}, "
                        f"now={e_now:.12g}, improve={improve:.3e} < {EARLY_STOP_TOL}"
                    )
                    stopped_early = True
                    break

        prev_mpack = mpack_out
        elapsed = time.time() - point_start
        if stopped_early:
            print(f"  Done (early-stopped) in {elapsed:.1f}s.  Checkpoint: {mpack_out}")
        else:
            print(f"  Done in {elapsed:.1f}s.  Checkpoint: {mpack_out}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All {total} points finished in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # worker 模式：参数为 alphaInt 子集（逗号分隔）
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        subset = [float(x) for x in sys.argv[2].split(",") if x.strip() != ""]
        main(alpha_subset=subset)
    else:
        main(alpha_subset=None)
