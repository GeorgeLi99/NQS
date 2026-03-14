#!/usr/bin/env python3
"""
批量训练 Long-range Ising RBM：蛇形遍历 (alphaInt, J) 网格 + 迁移学习。

功能概述
--------
在 (J, alphaInt) 参数网格上批量训练 RBMSymm 变分波函数，用于绘制相图。
采用蛇形遍历顺序，相邻参数点之间通过 checkpoint 迁移学习加速收敛。

训练模式
--------
1. 从头训练：第一个点或无可迁移 checkpoint 时，随机初始化，训练 N_ITER_FIRST 轮
2. 迁移学习：从上一个相邻点的 checkpoint 初始化，训练 N_ITER_TRANSFER 轮
3. 继续训练：若当前点已有 checkpoint，可按 config 选择 skip / 继续 500 轮 / 继续 1200 轮

输出路径
--------
phase_diagram/train/<PRECISION>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/
  - *.log：NetKet 训练日志（可被 parse_all_logs.py 解析）
  - *.mpack：Flax 序列化的模型 checkpoint
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any, cast

# 路径设置：确保可导入项目根目录及 config
sys.path.append("../../")
sys.path.append("../")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALPHA_INT_LIST, J_LIST,
    L, delta, Omega, PRECISION, alpha_rbm, key_cal, use_bias,
    N_ITER_FIRST, N_ITER_TRANSFER,
    N_ITER_CONTINUE_500, N_ITER_CONTINUE_1200,
    val_learning_rate, val_diagonal_shift,
    N_samples, n_chains_per_rank, n_discard_per_chain, chunk_size,
    param_subdir as _param_subdir, file_base as _file_base,
    EARLY_STOP_WINDOW, EARLY_STOP_TOL,
    USE_LR_SCHEDULE, LR_DECAY_ALPHA,
    ON_EXISTING_CHECKPOINT,
    OUTPUT_ROOT,
    USE_SR, SR_SOLVER, SR_SOLVER_TOL, SR_SOLVER_MAXITER, SR_SOLVER_RESTART,
)

# 输出根目录：优先环境变量 PHASE_DIAGRAM_OUTPUT_ROOT，其次 config.OUTPUT_ROOT，最后脚本所在目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
_output_root = os.environ.get("PHASE_DIAGRAM_OUTPUT_ROOT") or OUTPUT_ROOT or _script_dir
_output_root = os.path.abspath(os.path.expanduser(str(_output_root)))

# 在导入 JAX 前设置平台，确保使用 GPU
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jssl
import flax
import flax.serialization
import flax.linen as nn
import optax as opx
from functools import partial

from netket.operator.spin import sigmax, sigmaz
from netket.utils.group import PermutationGroup, Permutation

print("netket version:", nk.__version__)
print(f"Using platform: {jax.default_backend()}")

def _dtypes(precision: str) -> tuple:
    """根据精度字符串返回 (numpy dtype, jax dtype)。"""
    if precision == "complex64":
        return np.complex64, jnp.complex64
    return np.complex128, jnp.complex128

# 默认精度（config 指定），第一个点会临时切换为 complex128
dtype_np, dtype_jnp = _dtypes(PRECISION)

def make_optimizer(n_iter: int) -> tuple[Any, Any]:
    """
    根据 config 创建 SGD 优化器。返回 (optimizer, lr_schedule_fn)。
    lr_schedule_fn：接受 step 返回当前学习率的 callable；常数学习率时为 None。
    """
    if not USE_LR_SCHEDULE:
        return nk.optimizer.Sgd(learning_rate=float(val_learning_rate)), None

    lr_schedule = opx.cosine_decay_schedule(
        init_value=float(val_learning_rate),
        decay_steps=int(n_iter),
        alpha=float(LR_DECAY_ALPHA),
    )
    return nk.optimizer.Sgd(learning_rate=cast(Any, lr_schedule)), lr_schedule


def _energy_mean(driver: Any) -> float:
    """
    返回当前 step 的能量均值（尽量稳健地转成 python float）。
    driver.energy 可能是复数或 JAX DeviceArray，此处取实部。
    """
    e: Any = driver.energy.mean
    return float(np.real(np.asarray(e)))


def _train_dir(J: float, alpha_int: float) -> str:
    """
    返回当前参数点的训练输出目录，不存在则创建。
    路径格式：<output_root>/train/<PRECISION>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/
    """
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
    """返回当前参数点的 checkpoint 文件路径（.mpack）。"""
    return os.path.join(_train_dir(J, alpha_int), _file_base(J, alpha_int) + ".mpack")


def _log_base(J: float, alpha_int: float) -> str:
    """返回当前参数点的 log 文件前缀（不含扩展名，NetKet 会追加 .log）。"""
    return os.path.join(_train_dir(J, alpha_int), _file_base(J, alpha_int))


# ======================================================================
# 构建蛇形遍历顺序
# ======================================================================
def build_snake_order(alpha_list: list, j_list: list) -> list[tuple[float, float]]:
    """
    返回 [(J, alphaInt), ...] 的蛇形遍历列表。
    奇数行 J 反向，使相邻 alpha 的相邻 J 在参数空间中连续，利于迁移学习。
    例：alpha=[0.5,1.0], J=[0.5,1.0,1.5] → (0.5,0.5),(1.0,0.5),(1.5,0.5),(1.5,1.0),(1.0,1.0),(0.5,1.0)
    """
    order = []
    for idx, alpha in enumerate(alpha_list):
        js = j_list if idx % 2 == 0 else list(reversed(j_list))
        for j in js:
            order.append((j, alpha))
    return order


# ======================================================================
# 构建 Hamiltonian & observables（每换参数需重建）
# ======================================================================
def build_hamiltonian(hi, J_val: float, alpha_int: float, op_dtype=None):
    """
    长程横场 Ising 哈密顿量：
    H = (Ω/2) Σ_j σ^x_j - δ Σ_j σ^z_j + Σ_{i<j} (J/r^α) σ^z_i σ^z_j
    r 为周期边界下的最短距离。op_dtype 为算符矩阵的数据类型。
    """
    dt = op_dtype if op_dtype is not None else dtype_np
    Sigma_z = np.array([[1, 0], [0, -1]], dtype=dt)
    Sigma_x = np.array([[0, 1], [1, 0]], dtype=dt)

    H = nk.operator.LocalOperator(hi, dtype=dt)
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


def build_observables(hi, op_dtype=None):
    """
    构建观测量：
      Mx          = (1/L) Σ_j σ^x_j
      Mz          = (1/L) Σ_j σ^z_j
      Mz_AFM      = (1/L) Σ_j (-1)^j σ^z_j
      Mz_AFM_sq   = (1/L²) Σ_{i,j} (-1)^{i+j} σ^z_i σ^z_j  =  (Mz_AFM)²
                     即 staggered structure factor，等价于关联函数
                     Σ_j (-1)^j <σ^z_0 σ^z_j> / L 的全链求和。
                     热力学极限下 <Mz_AFM_sq> → (M_z^AFM)²。
                     波函数有平移对称性时 <Mz_AFM>=0，需看此量判断 AFM 序。
    """
    dt = op_dtype if op_dtype is not None else dtype_np
    Mx = nk.operator.LocalOperator(hi, dtype=dt)
    Mz = nk.operator.LocalOperator(hi, dtype=dt)
    Mz_AFM = nk.operator.LocalOperator(hi, dtype=dt)
    for j in range(L):
        Mx += (1 / L) * sigmax(hi, j)
        Mz += (1 / L) * sigmaz(hi, j)
        Mz_AFM += ((1 / L) * (-1) ** j) * sigmaz(hi, j)

    Mz_AFM_sq = Mz_AFM @ Mz_AFM

    return {"Mx": Mx, "Mz": Mz, "Mz_AFM": Mz_AFM, "Mz_AFM_sq": Mz_AFM_sq}


def _build_model(param_dtype_jnp, translation_group):
    """根据指定 param_dtype 创建 RBMSymm 模型。"""
    return nk.models.RBMSymm(
        alpha=alpha_rbm,
        param_dtype=param_dtype_jnp,
        use_hidden_bias=True,
        use_visible_bias=use_bias,
        symmetries=translation_group,
        kernel_init=nn.initializers.normal(stddev=0.01),
        hidden_bias_init=nn.initializers.normal(stddev=0.01),
    )


# ----------------------------------------------------------------------
# 平移对称性（与 RBMSymm/rbmsymm_long_range_ising.py 一致）
# ----------------------------------------------------------------------
def n_site_translation(L: int, n: int) -> list[int]:
    """L 格点链平移 n 格后的置换：site i → site (i+n) mod L。"""
    return [(i + n) % L for i in range(L)]


def build_translation_symmetries(L: int) -> PermutationGroup:
    """构建 L 个平移对称元（T(0)..T(L-1)），供 RBMSymm 的 DenseSymm 层使用，减少参数量。"""
    group_elems: list[Any] = [
        Permutation(np.array(n_site_translation(L, k), dtype=np.int32), name=f"T({k})")
        for k in range(L)
    ]
    return PermutationGroup(elems=group_elems, degree=L)


# ======================================================================
# Main training loop
# ======================================================================
def main(alpha_subset: list[float] | None = None):
    """
    主训练流程：蛇形遍历 (J, alphaInt)，每个点训练后保存 checkpoint，下一相邻点迁移学习。
    alpha_subset：若指定，仅训练该 alphaInt 子集（用于多 GPU 分片，当前已移除双卡逻辑）。
    """
    total_start = time.time()

    # 图与 Hilbert 空间：一维链，周期边界，自旋 1/2
    g = nk.graph.Chain(length=L, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    translation_group = build_translation_symmetries(L)
    print(f"RBMSymm: translation symmetries (L={L} elements)")

    model_cfg = _build_model(dtype_jnp, translation_group)

    # 采样器
    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=n_chains_per_rank, sweep_size=4 * L)

    # SR 预条件子（自然梯度）：USE_SR=True 时启用，显式配置 CG 求解器
    if USE_SR:
        _solver_kwargs: dict[str, Any] = {"tol": SR_SOLVER_TOL}
        if SR_SOLVER_MAXITER is not None:
            _solver_kwargs["maxiter"] = SR_SOLVER_MAXITER

        if SR_SOLVER == "gmres":
            _cg_solver = partial(jssl.gmres, **_solver_kwargs)
        else:
            _cg_solver = partial(jssl.cg, **_solver_kwargs)

        sr = nk.optimizer.SR(
            qgt=nk.optimizer.qgt.QGTJacobianPyTree,
            solver=_cg_solver,
            diag_shift=val_diagonal_shift,
            holomorphic=True,
            solver_restart=SR_SOLVER_RESTART,
        )
    else:
        sr = None

    # 打印参数量
    _vs_tmp = nk.vqs.MCState(sampler=sampler, model=model_cfg, n_samples=16)
    print(f"RBMSymm: alpha={alpha_rbm}, L={L}, PRECISION={PRECISION}, n_parameters={_vs_tmp.n_parameters}")
    del _vs_tmp

    alpha_list = ALPHA_INT_LIST if alpha_subset is None else alpha_subset
    order = build_snake_order(alpha_list, J_LIST)
    total = len(order)
    print(f"\n{'=' * 70}")
    print(f"Phase diagram: {total} sampling points (snake order)")
    print(f"  ALPHA_INT_LIST = {alpha_list}")
    print(f"  J_LIST         = {J_LIST}")
    print(f"  L={L}, delta={delta}, Omega={Omega}, PRECISION={PRECISION}")
    if USE_SR:
        print(f"  SR: ON (solver={SR_SOLVER}, tol={SR_SOLVER_TOL}, maxiter={SR_SOLVER_MAXITER}, restart={SR_SOLVER_RESTART})")
    else:
        print(f"  SR: OFF (pure SGD)")
    print(f"  N_ITER first={N_ITER_FIRST}, transfer={N_ITER_TRANSFER}")
    print(f"  ON_EXISTING_CHECKPOINT = {ON_EXISTING_CHECKPOINT!r}")
    print(f"{'=' * 70}\n")

    prev_mpack: str | None = None  # 上一相邻点的 checkpoint 路径，用于迁移学习

    for idx, (J_val, alpha_int) in enumerate(order):
        point_start = time.time()
        mpack_out = _mpack_path(J_val, alpha_int)
        checkpoint_exists = os.path.isfile(mpack_out)

        # 已有 checkpoint 且配置为 skip：直接跳过
        if checkpoint_exists and ON_EXISTING_CHECKPOINT == "skip":
            print(f"\n[{idx + 1}/{total}]  J={J_val}, alphaInt={alpha_int}  SKIP (checkpoint exists)")
            prev_mpack = mpack_out
            continue

        # 已有 checkpoint 且配置为 continue_500/continue_1200：加载自身 checkpoint，继续训练
        if checkpoint_exists:
            load_from = mpack_out
            if ON_EXISTING_CHECKPOINT == "continue_500":
                n_iter = N_ITER_CONTINUE_500
            elif ON_EXISTING_CHECKPOINT == "continue_1200":
                n_iter = N_ITER_CONTINUE_1200
            else:
                print(f"\n[{idx + 1}/{total}]  J={J_val}, alphaInt={alpha_int}  SKIP (ON_EXISTING_CHECKPOINT={ON_EXISTING_CHECKPOINT!r} 无效)")
                prev_mpack = mpack_out
                continue
            mode_str = "继续训练"
        else:
            # 无 checkpoint：从 prev_mpack 迁移学习，或从头训练
            load_from = prev_mpack
            n_iter = N_ITER_FIRST if load_from is None else N_ITER_TRANSFER
            mode_str = "迁移学习" if load_from else "从头训练"

        cur_precision = PRECISION
        cur_model = model_cfg
        cur_np, cur_jnp = _dtypes(PRECISION)

        print(f"\n[{idx + 1}/{total}]  J={J_val}, alphaInt={alpha_int}  ({mode_str}, {n_iter} iters, {cur_precision})")

        H = build_hamiltonian(hi, J_val, alpha_int, op_dtype=cur_np)
        obs = build_observables(hi, op_dtype=cur_np)

        # 变分态：每个点新建 MCState，随后从 load_from 加载参数
        vs = nk.vqs.MCState(
            sampler=sampler, model=cur_model,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size, n_samples=N_samples,
        )

        print(f"  Number of parameters: {vs.n_parameters}")

        # 加载 checkpoint；若结构不兼容（如旧版 plain RBM vs 当前 RBMSymm）则保持随机初始化
        loaded_ok = False
        if load_from is not None and os.path.isfile(load_from):
            try:
                with open(load_from, "rb") as f:
                    vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
                loaded_ok = True
                print(f"  Loaded checkpoint: {load_from}")
            except Exception as e:
                print(f"  WARNING: checkpoint 无法加载，从头训练: {e}")
        elif load_from is not None:
            print(f"  WARNING: checkpoint not found: {load_from}, training from scratch")

        opt, lr_schedule_fn = make_optimizer(n_iter)
        vmc = nk.VMC(hamiltonian=H, optimizer=opt, variational_state=vs, preconditioner=cast(Any, sr))

        out_base = _log_base(J_val, alpha_int)  # NetKet 会写入 out_base.log 和 out_base.mpack
        print(f"  Output: {out_base}.log")
        if USE_LR_SCHEDULE:
            print(f"  LR schedule: cosine decay, lr0={val_learning_rate}, alpha_end={LR_DECAY_ALPHA}, max_steps={n_iter}")
        else:
            print(f"  LR schedule: constant lr={val_learning_rate}")
        print(f"  Early stop: window={EARLY_STOP_WINDOW}, tol={EARLY_STOP_TOL}")

        # 预写测试：确保目录可写（避免 WSL/挂载盘权限问题导致 log 无法写入）
        train_dir = _train_dir(J_val, alpha_int)
        _test_file = os.path.join(train_dir, ".write_test")
        try:
            with open(_test_file, "w") as f:
                f.write("ok")
            os.remove(_test_file)
        except OSError as e:
            print(f"  WARNING: 目录不可写 {train_dir}: {e}")
            print(f"  建议: export PHASE_DIAGRAM_OUTPUT_ROOT=/path/to/writable/phase_diagram")

        energies: list[float] = []  # 记录每步能量，用于早停判据

        def early_stop_callback(step: int, log_data: dict, driver: Any) -> bool:
            """NetKet callback：返回 False 时停止训练。最近 EARLY_STOP_WINDOW 步能量改善 < tol 则早停。"""
            e_now = _energy_mean(driver)
            energies.append(e_now)
            if len(energies) <= EARLY_STOP_WINDOW:
                return True
            prev_best = float(np.min(energies[-(EARLY_STOP_WINDOW + 1) : -1]))
            delta_e = abs(prev_best - e_now)
            if delta_e < EARLY_STOP_TOL:
                print(
                    f"  Early stopping at iter {step}/{n_iter}: "
                    f"best(prev {EARLY_STOP_WINDOW})={prev_best:.12g}, "
                    f"now={e_now:.12g}, |ΔE|={delta_e:.3e} < {EARLY_STOP_TOL}"
                )
                return False
            return True

        # 单次 vmc.run 写入完整 log；callback 实现早停
        def _do_run():
            vmc.run(out=out_base, n_iter=n_iter, obs=obs, callback=early_stop_callback)

        # 若加载的 checkpoint 与当前模型结构不兼容（如 RBM vs RBMSymm），会触发 ScopeParamShapeError，此处重试从头训练
        try:
            _do_run()
        except Exception as e:
            err_msg = str(e)
            err_type = type(e).__name__
            if loaded_ok and ("ScopeParamShape" in err_type or "shape" in err_msg.lower() or "bias" in err_msg.lower()):
                print(f"  WARNING: checkpoint 与当前模型结构不兼容（如 RBM vs RBMSymm），从头训练")
                vs = nk.vqs.MCState(
                    sampler=sampler, model=cur_model,
                    n_discard_per_chain=n_discard_per_chain,
                    chunk_size=chunk_size, n_samples=N_samples,
                )
                vmc = nk.VMC(hamiltonian=H, optimizer=opt, variational_state=vs, preconditioner=cast(Any, sr))
                energies.clear()
                _do_run()
            else:
                raise

        prev_mpack = mpack_out  # 更新为当前点 checkpoint，供下一相邻点迁移学习
        elapsed = time.time() - point_start
        stopped_early = len(energies) < n_iter
        if stopped_early:
            print(f"  Done (early-stopped) in {elapsed:.1f}s.  Checkpoint: {mpack_out}")
        else:
            print(f"  Done in {elapsed:.1f}s.  Checkpoint: {mpack_out}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All {total} points finished in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
