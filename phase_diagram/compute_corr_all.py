#!/usr/bin/env python3
"""
批量补算 Mz_AFM_sq（staggered structure factor）并注入已有 .log 文件。

对 phase_diagram/train/<precision>/ 下的每个参数点：
  1. 加载 .mpack checkpoint → 构建 MCState
  2. 计算 <(Mz_AFM)²> = <σ^z staggered structure factor>
  3. 将结果作为新 key "Mz_AFM_sq" 写入对应的 .log JSON
  4. 重新生成 _run1_parsed.csv 和 _run1_summary.csv

适用于在引入 Mz_AFM_sq 观测量之前训练的模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import flax
import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from netket.utils.group import PermutationGroup, Permutation

from config import (
    ALPHA_INT_LIST, J_LIST,
    L, delta, Omega,
    alpha_rbm, key_cal, use_bias,
    N_samples, n_chains_per_rank, n_discard_per_chain, chunk_size,
    PRECISION as DEFAULT_PRECISION,
    param_subdir as _param_subdir,
    file_base as _file_base,
    OUTPUT_ROOT,
)

# 也导入 parse_all_logs 的解析函数，用于重新生成 CSV
from parse_all_logs import load_log, save_to_csv, save_summary_csv

print("netket version:", nk.__version__)
print(f"Using platform: {jax.default_backend()}")


def _dtypes(precision: str):
    if precision == "complex64":
        return np.complex64, jnp.complex64
    return np.complex128, jnp.complex128


def n_site_translation(L: int, n: int) -> list[int]:
    return [(i + n) % L for i in range(L)]


def build_translation_symmetries(L: int) -> PermutationGroup:
    group_elems: list[Any] = [
        Permutation(np.array(n_site_translation(L, k), dtype=np.int32), name=f"T({k})")
        for k in range(L)
    ]
    return PermutationGroup(elems=group_elems, degree=L)


def build_Mz_AFM_sq(hi):
    """构建 (Mz_AFM)² = [(1/L) Σ_j (-1)^j σ^z_j]² 算符。"""
    Mz_AFM = nk.operator.LocalOperator(hi, dtype=np.complex128)
    for j in range(L):
        Mz_AFM += ((1 / L) * (-1) ** j) * sigmaz(hi, j)
    return Mz_AFM @ Mz_AFM


def main():
    parser = argparse.ArgumentParser(description="批量补算 Mz_AFM_sq 并注入 log")
    parser.add_argument("-p", "--precision", default=DEFAULT_PRECISION,
                        choices=("complex64", "complex128"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出要处理的文件，不实际计算")
    args = parser.parse_args()

    dtype_np, dtype_jnp = _dtypes(args.precision)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.environ.get("PHASE_DIAGRAM_OUTPUT_ROOT") or OUTPUT_ROOT or script_dir
    output_root = os.path.abspath(os.path.expanduser(str(output_root)))

    # 构建一次就够：图、Hilbert、对称群、模型、采样器
    g = nk.graph.Chain(length=L, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    translation_group = build_translation_symmetries(L)
    model = nk.models.RBMSymm(
        alpha=alpha_rbm,
        param_dtype=dtype_jnp,
        use_hidden_bias=True,
        use_visible_bias=use_bias,
        symmetries=translation_group,
        kernel_init=nn.initializers.normal(stddev=0.01),
        hidden_bias_init=nn.initializers.normal(stddev=0.01),
    )
    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains_per_rank=n_chains_per_rank, sweep_size=4 * L
    )

    Mz_AFM_sq_op = build_Mz_AFM_sq(hi)

    # 收集要处理的点
    tasks = []
    for alpha_int in ALPHA_INT_LIST:
        for J_val in J_LIST:
            subdir = _param_subdir(J_val, alpha_int)
            base_dir = os.path.join(output_root, "train", args.precision, subdir)
            base_name = _file_base(J_val, alpha_int)
            log_path = os.path.join(base_dir, base_name + ".log")
            mpack_path = os.path.join(base_dir, base_name + ".mpack")
            if os.path.isfile(log_path) and os.path.isfile(mpack_path):
                tasks.append((J_val, alpha_int, log_path, mpack_path, base_dir, base_name))

    total = len(tasks)
    print(f"\nFound {total} points with both .log and .mpack\n")

    if args.dry_run:
        for J_val, alpha_int, log_path, mpack_path, *_ in tasks:
            print(f"  J={J_val}, alphaInt={alpha_int}")
        return

    ok, skip, fail = 0, 0, 0
    total_start = time.time()

    for idx, (J_val, alpha_int, log_path, mpack_path, base_dir, base_name) in enumerate(tasks):
        t0 = time.time()
        print(f"[{idx + 1}/{total}]  J={J_val}, alphaInt={alpha_int}", end="", flush=True)

        # 检查 log 中是否已有 Mz_AFM_sq
        try:
            data = load_log(log_path)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  FAIL (bad log): {e}")
            fail += 1
            continue

        if "Mz_AFM_sq" in data:
            print("  SKIP (already has Mz_AFM_sq)")
            skip += 1
            continue

        # 加载 checkpoint
        vs = nk.vqs.MCState(
            sampler=sampler, model=model,
            n_discard_per_chain=n_discard_per_chain,
            chunk_size=chunk_size, n_samples=N_samples,
        )
        try:
            with open(mpack_path, "rb") as f:
                vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
        except Exception as e:
            print(f"  FAIL (mpack load): {e}")
            fail += 1
            continue

        # 计算 <(Mz_AFM)²>
        try:
            stat = vs.expect(Mz_AFM_sq_op)
            val_real = float(np.real(stat.mean))
            val_imag = float(np.imag(stat.mean))
            val_var = float(np.real(stat.variance))
            val_sigma = float(np.real(stat.error_of_mean))
        except Exception as e:
            print(f"  FAIL (expect): {e}")
            fail += 1
            continue

        # 构造与其他观测量相同格式的条目，复制最后一个迭代的 iters
        n_iters = len(data["Energy"]["iters"])
        iters_list = data["Energy"]["iters"]

        # 只填最后一个点有真实值，其余填 NaN（因为之前训练时未记录）
        nan_list = [float("nan")] * n_iters
        real_list = nan_list.copy()
        imag_list = nan_list.copy()
        var_list = nan_list.copy()
        sigma_list = [None] * n_iters  # type: ignore
        r_hat_list = [None] * n_iters  # type: ignore

        real_list[-1] = val_real
        imag_list[-1] = val_imag
        var_list[-1] = val_var
        sigma_list[-1] = val_sigma  # type: ignore

        data["Mz_AFM_sq"] = {
            "iters": list(iters_list),
            "Mean": {"real": real_list, "imag": imag_list},
            "Variance": var_list,
            "Sigma": sigma_list,
            "R_hat": r_hat_list,
        }

        # 写回 log
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # 重新生成 parsed CSV 和 summary CSV
        parsed_path = os.path.join(base_dir, f"{base_name}_run1_parsed.csv")
        summary_path = os.path.join(base_dir, f"{base_name}_run1_summary.csv")
        save_to_csv(data, parsed_path)
        save_summary_csv(data, log_path, summary_path)

        elapsed = time.time() - t0
        print(f"  OK  Mz_AFM_sq={val_real:.8f} ± {val_sigma:.2e}  ({elapsed:.1f}s)")
        ok += 1

    total_elapsed = time.time() - total_start
    print(f"\nDone: {ok} computed, {skip} skipped, {fail} failed  ({total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
