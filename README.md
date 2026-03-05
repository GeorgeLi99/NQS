# NQS：Rydberg 链的神经量子态

基于 **神经量子态 (Neural Quantum States, NQS)** 与 **变分蒙特卡罗 (VMC)**，研究一维 Rydberg 原子链的长程横场 Ising 模型基态。项目将 NQS 与 MPS/DMRG 对比，并系统改变相互作用指数 α，探索何时 NQS 更具优势。

详细物理背景与任务说明见 [NQS project.md](NQS%20project.md)。

---

## 项目结构

```
0_nqs/
├── README.md              # 本文件：介绍与环境配置
├── requirements.txt       # Python 依赖列表
├── NQS project.md        # 课程/项目说明（模型、任务、参考文献）
├── rydberg_*.log, rydberg_*.mpack   # （若存在）根目录下为 complex64 精度运行所生成，见下方说明
└── rydberg_chain/
    ├── rydberg_nqs_starter.py   # NQS 入门脚本；TRAIN_SUBDIR 指定精度子目录，LOAD_CHECKPOINT 可恢复
    ├── parse_vmc_log.py         # 解析 .log → *_runN_parsed.csv / *_runN_summary.csv（不覆盖，-r 指定 N）
    ├── merge_vmc_csvs.py        # 将多次训练 *_run*_*.csv 合并为 *_merged_parsed.csv、*_merged_summary.csv
    ├── how_to_load_model.py     # 从 .mpack 加载参数的示例
    ├── Fig_Convergence_Obs.py   # 收敛与观测量画图（读合并后的 CSV，输出到 figure/）
    ├── Fig_Convergence_Obs_compare.py   # complex128 vs complex64 对比图（读各精度下合并 CSV）
    ├── figure/                  # 绘图输出目录（PDF/SVG）
    └── train/                   # 训练输出目录，按精度分子目录
        ├── complex128/          # 双精度运行输出（默认）
        │   ├── rydberg_L*.log, rydberg_L*.mpack
        │   ├── *_runN_parsed.csv, *_runN_summary.csv   # 单次训练解析结果
        │   └── *_merged_parsed.csv, *_merged_summary.csv   # merge_vmc_csvs 合并结果
        └── complex64/           # 单精度运行输出
            └── （同上）
```

---

## 环境要求

- **Python**：≥ 3.9（推荐 3.10 或 3.11）
- **主要依赖**（`requirements.txt` 中已锁定实测版本）：
  - NetKet 3.19.2
  - JAX / jaxlib 0.4.38
  - Flax 0.10.4、Optax 0.2.5
  - NumPy 2.0.2
- **运行环境**：推荐在 **WSL2 (Windows Subsystem for Linux)** 下配置 Python 环境，便于与 Linux 生态一致，并可选 GPU（CUDA on WSL）

---

## WSL 中 Python 环境配置

以下均在 **WSL2** 的终端（如 `wsl` 或 `Ubuntu`）中执行。

### 1. 确认 WSL 与 Python

```bash
# 查看 WSL 版本（建议为 2）
wsl --version

# 进入 WSL 后，检查是否已有 Python3
python3 --version
which python3
```

若未安装 Python：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. 创建项目目录与虚拟环境

**方式 A：使用 `venv`（推荐，轻量）**

```bash
cd ~/path/to/0_nqs   # 请替换为你本机的项目根目录路径

# 创建虚拟环境（可放在项目内 .venv，或用户目录下如 nqs_wsl）
python3 -m venv .venv
# 若希望在家目录下单独建环境，例如：
# python3 -m venv ~/nqs_wsl

# 激活虚拟环境（根据上一步实际路径）
source .venv/bin/activate
# 若环境在用户目录： source ~/nqs_wsl/bin/activate

# 提示符前会出现 (.venv) 或 (nqs_wsl)，表示已激活
```

**方式 B：使用 Conda（若已安装 Anaconda/Miniconda）**

```bash
cd ~/path/to/0_nqs

# 创建独立环境，指定 Python 版本
conda create -n nqs python=3.11 -y

# 激活
conda activate nqs
```

### 3. 升级 pip 与安装依赖

**重要**：先升级 pip（NetKet 等依赖需要较新的 pip）。

```bash
# 在已激活的 .venv 或 conda 环境中执行
pip install --upgrade pip setuptools wheel
```

**仅 CPU：**

```bash
pip install -r requirements.txt
```

**若在 WSL 中使用 NVIDIA GPU（CUDA on WSL）：**

先确认驱动与 CUDA（在 WSL 内 `nvidia-smi` 可用）。**先装带 CUDA 的 JAX**（会拉取 jaxlib 等），再装本项目的 requirements，避免版本冲突：

```bash
# 先安装 JAX GPU 版（常见为 CUDA 12，与实测环境一致）
pip install --upgrade "jax[cuda12]"

# 再安装其余包（与 requirements.txt 中版本一致：NetKet 3.19.2, Flax 0.10.4 等）
pip install -r requirements.txt
```

若为 CUDA 11：

```bash
pip install --upgrade "jax[cuda11_local]"
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python3 -c "
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax
from flax import linen as nn
print('NumPy:', np.__version__)
print('JAX:', jax.__version__)
print('NetKet:', nk.__version__)
print('JAX backend:', jax.default_backend())
print('Devices:', jax.devices())
"
```

无报错即表示环境可用。在实测 WSL 环境中应看到类似：`JAX: 0.4.38`、`NetKet: 3.19.2`、`JAX backend: gpu`（若已装 `jax[cuda12]`）。

### 5. 运行入门脚本

```bash
# 确保在项目根目录且已激活环境
cd ~/path/to/0_nqs
source .venv/bin/activate   # 或 source ~/nqs_wsl/bin/activate / conda activate nqs

python3 rydberg_chain/rydberg_nqs_starter.py
```

脚本会打印 JAX/NetKet 版本与设备信息，并进行一次小规模 NQS 优化。运行完成后，在 **`rydberg_chain/train/<precision>/`** 下会生成（`<precision>` 由脚本内 `TRAIN_SUBDIR` 决定，与模型精度一致：默认 **complex128**，单精度时改为 **complex64**）：

- **`rydberg_L{L}_delta{δ}_Rb{Rb}_alpha{α}.log`**：每步的能量、方差、观测量（如 Mx、Mz、Ntot）等文本日志。
- **`rydberg_L{L}_delta{δ}_Rb{Rb}_alpha{α}.mpack`**：NetKet 的二进制检查点，便于用 `LOAD_CHECKPOINT` 恢复训练。

解析 log 并保存 CSV 时，路径与上述一致：不指定 log 时默认读 **`train/complex128/`** 下默认文件名，可用 `-p complex64` 改为读 **`train/complex64/`**。解析结果默认保存为 **`*_runN_parsed.csv`** 与 **`*_runN_summary.csv`**（N 为第几次训练，自动递增不覆盖）；可用 `-r N` 指定 N。

**说明**：训练与解析的读写路径统一为 **`rydberg_chain/train/complex128/`** 或 **`rydberg_chain/train/complex64/`**；若在项目根或 `rydberg_chain/` 下存在旧版 log/mpack，可忽略或移至对应 `train/<precision>/`。

### 6. 解析、合并与画图（可选）

多次训练（含从 checkpoint 微调续训）时，建议流程：

1. **解析**：每次训练得到新 log 后运行  
   `python3 rydberg_chain/parse_vmc_log.py`（或 `-p complex64`），得到 `*_run1_*`、`*_run2_*`、…，不覆盖。
2. **合并**：将同一目录下多次 run 的 CSV 合并为一条长轨迹（含 `global_iter` 连续编号）：  
   `python3 rydberg_chain/merge_vmc_csvs.py [目录]`  
   默认目录为 `rydberg_chain/train/complex128`，生成 **`*_merged_parsed.csv`** 与 **`*_merged_summary.csv`**。
3. **画图**：  
   - **单精度收敛图**：`python3 rydberg_chain/Fig_Convergence_Obs.py`  
     读取合并后的 CSV（若不存在则回退到 `*_run1_*` 或 `*_parsed.csv`），左图相对误差、右图观测量，输出到 **`rydberg_chain/figure/`**。  
   - **complex128 vs complex64 对比图**：`python3 rydberg_chain/Fig_Convergence_Obs_compare.py`  
     读取 `train/complex128/` 与 `train/complex64/` 下合并后的 CSV，在同一张图左右两子图中对比两条精度曲线。

画图脚本优先使用 **`*_merged_*`**，便于把多次训练/微调视为一条连续曲线（横轴 `global_iter`）。

---

## 包管理说明

### 使用 requirements.txt

- **安装当前项目依赖**：`pip install -r requirements.txt`
- **冻结当前环境**（便于复现）：  
  `pip freeze > requirements-full.txt`  
  （可选，一般用 `requirements.txt` 做“最小依赖”即可）
- **添加新依赖**：在 `requirements.txt` 中增加一行（可带版本，如 `package>=x.y`），然后重新执行 `pip install -r requirements.txt`

### 可选：MPI 并行

若需多进程运行（如 `mpirun -np 4 python3 rydberg_nqs_starter.py`），在 WSL 中可安装 OpenMPI 与 mpi4py（实测版本 4.1.1）：

```bash
sudo apt install libopenmpi-dev openmpi-bin
pip install mpi4py==4.1.1
```

并在 `rydberg_nqs_starter.py` 中按注释启用 MPI 相关代码。

### 虚拟环境常用命令

| 操作           | venv                    | Conda        |
|----------------|-------------------------|--------------|
| 激活           | `source .venv/bin/activate` | `conda activate nqs` |
| 退出           | `deactivate`            | `conda deactivate`   |
| 查看已装包     | `pip list`              | `conda list`         |
| 安装包         | `pip install <包名>`    | `conda install <包名>` 或 `pip install <包名>` |

---

## 项目任务概览（来自 NQS project.md）

1. **Task 3.1**：小系统精确对角化 (ED)，作为基准。
2. **Task 3.2**：在同一小系统上对比 NQS 与 DMRG，验证收敛与精度。
3. **Task 3.3**：固定其他参数，改变相互作用指数 α，比较 NQS 与 DMRG 在 α &le; 2 与 α &gt; 2 下的表现（核心任务）。
4. **Task 3.4**：在 NQS 占优的 α 下，绘制 (δ̃, J) 相图。

ED 可在 NetKet 内用精确求解器完成，或使用 QuSpin 等；DMRG 可使用 ITensor、TeNPy 等。

---

## 参考

- 模型与公式、任务细节、参考文献：见 [NQS project.md](NQS%20project.md)。
- [NetKet 文档](https://www.netket.org/)
- [JAX 文档](https://jax.readthedocs.io/)
