"""
phase_diagram 全局配置：所有脚本从此处读取参数，修改一处即可全局生效。
"""

# ======================================================================
# 采样网格（手动编辑这两个列表来增减采样点）
# ======================================================================
ALPHA_INT_LIST = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
J_LIST = [0.5, 1.0, 1.5, 2.0]

# ======================================================================
# 物理 / 系统参数
# ======================================================================
L = 32
delta = 0.0
Omega = 2.0

# ======================================================================
# RBM 超参
# ======================================================================
alpha_rbm = 4
key_cal = 1
use_bias = False

# ======================================================================
# 训练超参
# ======================================================================
PRECISION = "complex128"
N_ITER_FIRST = 1800      # 第一个采样点（无 checkpoint 可用）
N_ITER_TRANSFER = 1200   # 迁移学习的后续点

val_learning_rate = 0.0025
val_diagonal_shift = 0.0001
N_samples = 1024 * 32
n_chains_per_rank = 512
n_discard_per_chain = 32
chunk_size = 1024 * 8

# 早停（能量收敛判据）
EARLY_STOP_WINDOW = 50          # 最近多少个回合构成窗口
EARLY_STOP_TOL = 1e-5            # 若窗口内能量改善 < tol 则早停

# 学习率调度（cosine decay）
USE_LR_SCHEDULE = True           # False 则使用常数学习率 val_learning_rate
LR_DECAY_ALPHA = 0.1             # cosine decay 末端比例（末端 lr = lr0 * alpha）

# 多 GPU（两张卡并行跑不同 alphaInt 行）
USE_TWO_GPUS = True             # True: 使用两张 GPU 并行；False: 单卡串行

# 输出目录根路径（用于云端无写权限时重定向输出）
# - None：默认写入 phase_diagram/ 目录下（即与脚本同级的 train/figure）
# - 或者给一个可写路径，比如 "/scratch/$USER/nqs/phase_diagram"
OUTPUT_ROOT = None

# ======================================================================
# 路径约定（由参数派生，一般无需改）
# ======================================================================

def param_subdir(J: float, alpha_int: float) -> str:
    return f"L{L}_J{J}_delta{delta}_alphaInt{alpha_int}"


def file_base(J: float, alpha_int: float) -> str:
    return f"rbm_LongIsing_L={L}_J={J}_delta={delta}_alphaInt={alpha_int}_alpha={alpha_rbm}_Cal{key_cal}"
