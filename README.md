# NQS：神经量子态（Rydberg 链与长程 Ising）

基于 **神经量子态 (Neural Quantum States, NQS)** 与 **变分蒙特卡罗 (VMC)**，使用 **Stochastic Reconfiguration (SR)** 预条件子，研究一维 Rydberg 原子链与长程横场 Ising 模型的基态。项目将 NQS 与 MPS/DMRG 对比，并系统改变相互作用指数 α，探索何时 NQS 更具优势。

详细物理背景与任务说明见 [NQS project.md](NQS%20project.md)。

---

## 项目结构

```
0_nqs/
├── README.md              # 本文件：介绍与环境配置
├── requirements.txt       # Python 依赖列表
├── NQS project.md        # 课程/项目说明（模型、任务、参考文献）
├── rydberg_chain/         # Rydberg 原子链 NQS（观测量 Mx, Mz, Ntot）
│   ├── rydberg_nqs_starter.py   # NQS 训练；PRECISION=complex64|complex128，LOAD_CHECKPOINT 可恢复
│   ├── parse_vmc_log.py         # 解析 .log → *_runN_parsed.csv / *_runN_summary.csv（-p 精度，-r 指定 N）
│   ├── merge_vmc_csvs.py        # 合并多次训练 CSV（--precision, --name）
│   ├── how_to_load_model.py     # 从 .mpack 加载参数示例
│   ├── Fig_Convergence_Obs.py   # 收敛与观测量（-p 精度，输出到 figure/）
│   ├── Fig_Convergence_Obs_compare.py   # complex128 vs complex64 对比
│   ├── figure/                  # 绘图输出（PDF/SVG）
│   └── train/
│       ├── complex128/          # 双精度 .log、.mpack、*_parsed.csv、*_merged_*.csv
│       └── complex64/           # 单精度（同上）
└── long_range_ising/      # 长程横场 Ising 模型 NQS（观测量 Mx, Mz, Mz_AFM，含 sigma 列）
    ├── rbm_long_range_ising.py  # RBM+VMC 训练；PRECISION、LOAD_CHECKPOINT、delta 等，文件名含 delta
    ├── parse_vmc_log.py         # 解析 .log，输出含 sigma_Mx/Mz/Mz_AFM（观测量方差）
    ├── merge_vmc_csvs.py        # 合并 CSV（默认 --precision complex64，--name 与训练脚本一致）
    ├── how_to_load_model.py     # 从 train/<precision>/ 加载 .mpack 示例
    ├── Fig_Convergence_Obs.py   # 收敛与 |Mx|、|Mz|、|Mz_AFM|（可选误差带）
    ├── Fig_Convergence_Obs_compare.py   # 双精度对比 + Mz_AFM
    ├── figure/
    └── train/
        ├── complex128/
        └── complex64/
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

**Rydberg 链：**

```bash
cd ~/path/to/0_nqs
source .venv/bin/activate   # 或 conda activate nqs

python3 rydberg_chain/rydberg_nqs_starter.py
```

运行完成后在 **`rydberg_chain/train/<precision>/`** 下生成 `.log` 与 `.mpack`。脚本内 **`PRECISION`**（complex64/complex128）决定精度与子目录；**`LOAD_CHECKPOINT`** 可设为同目录下 `.mpack` 路径以恢复训练。

**长程 Ising（long_range_ising）：**

```bash
python3 long_range_ising/rbm_long_range_ising.py
```

输出在 **`long_range_ising/train/<precision>/`**，文件名含 **`delta`**（如 `rbm_LongIsing_L=16_J=1.0_delta=0_alphaInt=2.0_...`），便于区分不同 δ。观测量为 **Mx、Mz、Mz_AFM**（反铁磁序参量 \(M_z^{\mathrm{AFM}}=(1/L)\sum_j (-1)^j\langle\sigma^z_j\rangle\)）。脚本内 **`PRECISION`**、**`LOAD_CHECKPOINT`**（True/False）、**`delta`** 等可调。

### 6. 解析、合并与画图（可选）

两模块流程一致：解析 log → 合并多次 run 的 CSV → 画收敛与观测量。

| 步骤 | rydberg_chain | long_range_ising |
|------|----------------|------------------|
| **解析** | `python3 rydberg_chain/parse_vmc_log.py`（可选 `-p complex64`） | `python3 long_range_ising/parse_vmc_log.py`（可选 `-p complex64`，默认基名与训练输出一致） |
| **合并** | `python3 rydberg_chain/merge_vmc_csvs.py` 或 `-p complex64` | `python3 long_range_ising/merge_vmc_csvs.py` 或 `-p complex64` |
| **单精度图** | `python3 rydberg_chain/Fig_Convergence_Obs.py -p complex64` | `python3 long_range_ising/Fig_Convergence_Obs.py -p complex64` |
| **双精度对比图** | `python3 rydberg_chain/Fig_Convergence_Obs_compare.py` | `python3 long_range_ising/Fig_Convergence_Obs_compare.py` |

- 解析得到 **`*_runN_parsed.csv`**、**`*_runN_summary.csv`**（N 自动递增；`-r N` 可指定）。long_range_ising 的 parsed CSV 含 **sigma_Mx、sigma_Mz、sigma_Mz_AFM**，便于查看观测量方差。
- 合并得到 **`*_merged_parsed.csv`**、**`*_merged_summary.csv`**（含 `run`、`global_iter`）。
- 图保存到各模块下的 **`figure/`**；long_range_ising 图含 \|Mx\|、\|Mz\|、\|Mz_AFM\| 及可选误差带。

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
