# 从 train/<precision>/ 下的 .mpack 加载 Long-range Ising RBM 参数
# 与 rydberg_chain/how_to_load_model.py 相同逻辑，路径为 long_range_ising/train/complex64 或 complex128

import os
import netket as nk
import flax
import flax.serialization

# 在定义 vs (MCState) 之后，从文件加载参数示例：
#
# _script_dir = os.path.dirname(os.path.abspath(__file__))
# PRECISION = "complex64"  # 或 "complex128"
# load_name = os.path.join(_script_dir, "train", PRECISION, "rbm_LongIsing_L=16_J=1.0_alphaInt=2.0_alpha=4_Cal1.mpack")
# if os.path.isfile(load_name):
#     with open(load_name, "rb") as f:
#         vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
#     print("Loaded parameters from:", load_name)
