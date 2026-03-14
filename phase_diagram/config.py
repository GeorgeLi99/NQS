"""
phase_diagram 全局配置：所有脚本从此处读取参数，修改一处即可全局生效。
"""

# ======================================================================
# 采样网格（手动编辑这两个列表来增减采样点）
# ======================================================================
ALPHA_INT_LIST = [0.5, 0.75, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
J_LIST = [2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5]

# ======================================================================
# 物理 / 系统参数
# ======================================================================
L = 32
delta = 0.0
Omega = 2.0

# ======================================================================
# RBM 超参
# ======================================================================
alpha_rbm = 12
key_cal = 1
use_bias = False

# ======================================================================
# 训练超参
# ======================================================================
PRECISION = "complex64"
N_ITER_FIRST = 1000      # 第一个采样点（无 checkpoint 可用）
N_ITER_TRANSFER = 1000   # 迁移学习的后续点

val_learning_rate = 0.001
val_diagonal_shift = 0.0001
N_samples = 1024 * 32
n_chains_per_rank = 512
n_discard_per_chain = 32
chunk_size = 1024 * 8

# 已有 checkpoint 时的行为（当前模型 mpack 已存在）
# - "skip": 跳过，不训练
# - "continue_500": 加载该 checkpoint，继续训练 N_ITER_CONTINUE_500 轮
# - "continue_1200": 加载该 checkpoint，继续训练 N_ITER_CONTINUE_1200 轮
ON_EXISTING_CHECKPOINT = "continue_500"
N_ITER_CONTINUE_500 = 500
N_ITER_CONTINUE_1200 = 1200

# 早停（能量收敛判据）
EARLY_STOP_WINDOW = 200          # 最近多少个回合构成窗口
EARLY_STOP_TOL = 5e-5            # 若窗口内能量改善 < tol 则早停

# SR（随机重整化 / 自然梯度）
USE_SR = True                   # True: 使用 SR 预条件（收敛快但每步慢）；False: 纯 SGD（每步快但收敛慢）

# SR 线性求解器（共轭梯度法，仅 USE_SR=True 时生效）
SR_SOLVER = "cg"                 # "cg": 共轭梯度法（默认），"gmres": 广义最小残差法
SR_SOLVER_TOL = 1e-4             # CG 收敛容差（越大越快但近似误差越大；推荐 1e-4 ~ 1e-6）
SR_SOLVER_MAXITER = 20           # CG 最大迭代次数（越小越快；None 表示不限）
SR_SOLVER_RESTART = False        # True: 每步从零开始；False: 用上一步的解作为初始猜测（更快）

# 学习率调度（cosine decay）
USE_LR_SCHEDULE = True           # False 则使用常数学习率 val_learning_rate
LR_DECAY_ALPHA = 0.1             # cosine decay 末端比例（末端 lr = lr0 * alpha）

# 多 GPU（两张卡并行跑不同 alphaInt 行）
USE_TWO_GPUS = False             # True: 使用两张 GPU 并行；False: 单卡串行

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
