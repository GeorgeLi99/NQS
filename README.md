# NQS：神经量子态（Rydberg 链与长程 Ising）

基于 **神经量子态 (Neural Quantum States, NQS)** 与 **变分蒙特卡罗 (VMC)**，使用 **Stochastic Reconfiguration (SR)** 预条件子，研究一维 Rydberg 原子链与长程横场 Ising 模型的基态。项目将 NQS 与 MPS/DMRG 对比，并系统改变相互作用指数 α，探索何时 NQS 更具优势。

详细物理背景与任务说明见 [NQS project.md](NQS%20project.md)。

---

## 项目结构

两套脚本的**路径约定一致**：训练产生的 `.log`、`.mpack`（checkpoint）与解析/合并得到的 CSV 均按**参数子目录**存放，便于同一组参数（L、δ、α 等）的 log、checkpoint、CSV 同处一目录。

```
0_nqs/
├── README.md              # 本文件：介绍与环境配置
├── requirements.txt      # Python 依赖列表
├── pyrightconfig.json     # 类型检查（phase_diagram 导入 config 用）
├── NQS project.md         # 课程/项目说明（模型、任务、参考文献）
├── ED/                    # 精确对角化（QuSpin Lanczos）基准
│   ├── ground_state_ising_lanczos.py   # 长程 Ising 基态：H = (Ω/2)Σσ^x - δΣσ^z + Σ_{i<j}(J/r^α)σ^z_iσ^z_j
│   └── result/            # 结果 CSV，按参数命名：ising_L{L}_J{J}_alpha{alpha}_delta{delta}_h{h}.csv
├── rydberg_chain/         # Rydberg 原子链 NQS（观测量 Mx, Mz, Ntot）
│   ├── rydberg_nqs_starter.py   # NQS 训练；输出到 train/<precision>/<param_subdir>/
│   ├── parse_vmc_log.py         # 解析 .log → *_runN_parsed.csv / *_runN_summary.csv
│   ├── merge_vmc_csvs.py        # 合并多次训练 CSV
│   ├── how_to_load_model.py     # 从 .mpack 加载参数示例
│   ├── Fig_Convergence_Obs.py   # 收敛与观测量（单精度）
│   ├── Fig_Convergence_Obs_compare.py   # complex128 vs complex64 对比
│   ├── figure/                  # 绘图输出（PDF/SVG）
│   └── train/
│       ├── complex128/
│       │   └── L{L}_Rb{Rb}_delta{delta}_alpha{alpha}/   # 例：L16_Rb1.0_delta0.5_alpha6.0
│       │       ├── rydberg_*.log, rydberg_*.mpack, *_parsed.csv, *_merged_*.csv
│       └── complex64/
│           └── L{L}_Rb{Rb}_delta{delta}_alpha{alpha}/
├── long_range_ising/      # 长程横场 Ising 模型 NQS（观测量 Mx, Mz, Mz_AFM）
│   ├── rbm_long_range_ising.py  # RBM+VMC 训练；输出到 train/<precision>/<param_subdir>/
│   ├── parse_vmc_log.py         # 解析 .log，输出含 sigma_Mx/Mz/Mz_AFM
│   ├── merge_vmc_csvs.py        # 合并 CSV
│   ├── how_to_load_model.py     # 从 .mpack 加载参数示例
│   ├── Fig_Convergence_Obs.py   # 收敛与 |Mx|、|Mz|、|Mz_AFM|
│   ├── Fig_Convergence_Obs_compare.py   # 双精度对比
│   ├── figure/
│   └── train/
│       ├── complex128/
│       │   └── L{L}_J{J}_delta{delta}_alphaInt{alpha}/   # 例：L16_J1.0_delta0.0_alphaInt2.0
│       │       ├── rbm_LongIsing_*.log, *.mpack, *_parsed.csv, *_merged_*.csv
│       └── complex64/
│           └── L{L}_J{J}_delta{delta}_alphaInt{alpha}/
├── RBMSymm/               # 带对称性的 RBM（长程 Ising）
│   └── rbmsymm_long_range_ising.py   # 平移对称 RBM；超参数在文件顶部
├── phase_diagram/         # 长程 Ising 相图流水线（网格训练 + 热力图）
│   ├── config.py          # 全局配置：ALPHA_INT_LIST, J_LIST, L, delta, 训练超参等
│   ├── run_phase_diagram.py      # 蛇形遍历 (J, alphaInt) 网格 + 迁移学习
│   ├── parse_all_logs.py        # 批量解析 .log → CSV
│   ├── plot_convergence.py      # 每点收敛图，保存到对应参数子目录
│   ├── plot_phase_diagram.py    # |Mx|、|Mz_AFM| 热力图 + 汇总 CSV
│   ├── figure/            # 相图 PDF/SVG
│   └── train/
│       └── complex64/
│           └── L{L}_J{J}_delta{delta}_alphaInt{alpha}/   # 每点 log、mpack、CSV、收敛图
└── ...
```

- **训练脚本**（`rydberg_nqs_starter.py`、`rbm_long_range_ising.py`）：根据当前参数生成 `param_subdir`，将 **.log** 与 **.mpack**（checkpoint）写入 `train/<PRECISION>/<param_subdir>/`。
- **解析 / 合并 / 画图**（`parse_vmc_log.py`、`merge_vmc_csvs.py`、`Fig_Convergence_Obs*.py`）：通过命令行参数 `--L`、`--delta`、`--alpha`（以及 rydberg 的 `--Rb`、long_range_ising 的 `--J`）确定同一组参数，默认读写路径为 `train/<precision>/<param_subdir>/`，与训练输出一致。

---

## 环境要求

- **Python**：≥ 3.9（推荐 3.10 或 3.11）
- **主要依赖**（`requirements.txt` 中已锁定实测版本）：
  - NetKet 3.19.2
  - JAX / jaxlib 0.4.38
  - Flax 0.10.4、Optax 0.2.5
  - NumPy 2.0.2
- **运行环境**：推荐在 **WSL2 (Windows Subsystem for Linux)** 下配置 Python 环境，便于与 Linux 生态一致，并可选 GPU（CUDA on WSL）
- **ED 脚本**（`ED/ground_state_ising_lanczos.py`）依赖 **QuSpin**；若仅运行 NQS 与相图流水线，可不安装

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

运行完成后在 **`rydberg_chain/train/<precision>/L{L}_Rb{Rb}_delta{delta}_alpha{alpha}/`** 下生成 `.log` 与 `.mpack`（例：默认参数为 `L16_Rb1.0_delta0.5_alpha6.0`）。脚本内 **`PRECISION`**、**`L`**、**`Rb`**、**`delta`**、**`alpha_interaction`** 决定子目录与文件名；**`LOAD_CHECKPOINT`** 指向同目录下 `.mpack` 以恢复训练。

**长程 Ising（long_range_ising）：**

```bash
python3 long_range_ising/rbm_long_range_ising.py
```

输出在 **`long_range_ising/train/<precision>/L{L}_J{J}_delta{delta}_alphaInt{alpha}/`**（例：默认 `L16_J1.0_delta0.0_alphaInt2.0`），文件名含 **`delta`**（如 `rbm_LongIsing_L=16_J=1.0_delta=0.0_alphaInt=2.0_...`）。观测量为 **Mx、Mz、Mz_AFM**（反铁磁序参量 \(M_z^{\mathrm{AFM}}=(1/L)\sum_j (-1)^j\langle\sigma^z_j\rangle\)）。脚本内 **`PRECISION`**、**`LOAD_CHECKPOINT`**、**`delta`**、**`L`**、**`J`**、**`alpha_interaction`** 等可调。

**精确对角化（ED）基准：**

```bash
python3 ED/ground_state_ising_lanczos.py
```

哈密顿量 \(H = (\Omega/2)\sum_i \sigma_i^x - \tilde{\delta}\sum_i \sigma_i^z + \sum_{i<j} (J/r_{ij}^\alpha) \sigma_i^z \sigma_j^z\)，参数在脚本顶部：**L、J、alpha、delta、h**（其中 h 对应 Ω/2）。结果写入 **`ED/result/`**，CSV 按参数命名：`ising_L{L}_J{J}_alpha{alpha}_delta{delta}_h{h}.csv`，含基态能量、AFM z 磁化、横磁化等。

**相图流水线（phase_diagram）：**

所有网格与训练配置在 **`phase_diagram/config.py`** 中统一修改（如 `ALPHA_INT_LIST`、`J_LIST`、`L`、`delta` 等）。在 `phase_diagram/` 目录下依次执行：

```bash
cd phase_diagram
python3 run_phase_diagram.py    # 蛇形遍历网格 + 迁移学习
python3 parse_all_logs.py       # 批量解析 .log → CSV
python3 plot_convergence.py     # 每点收敛图
python3 plot_phase_diagram.py   # |Mx|、|Mz_AFM| 热力图 → figure/
```

**对称 RBM（RBMSymm）：**

```bash
python3 RBMSymm/rbmsymm_long_range_ising.py
```

长程 Ising 的平移对称 RBM；**L、J、alpha_interaction、alpha_rbm、学习率、采样数等超参数均在文件最前部**，按需修改即可。

### 6. 解析、合并与画图（可选）

两模块流程一致：解析 log → 合并多次 run 的 CSV → 画收敛与观测量。**读写路径与训练一致**：均按参数子目录 `train/<precision>/<param_subdir>/`，通过 `--L`、`--delta`、`--alpha` 等指定要处理的参数组（rydberg 另加 `--Rb`，long_range_ising 另加 `--J`）；不指定时使用各脚本顶部的默认超参数。

| 步骤 | rydberg_chain | long_range_ising |
|------|----------------|------------------|
| **解析** | `python3 rydberg_chain/parse_vmc_log.py`<br>可选：`-p complex64`，`--L 16 --Rb 1.0 --delta 0.5 --alpha 6` | `python3 long_range_ising/parse_vmc_log.py`<br>可选：`-p complex64`，`--L 16 --J 1.0 --delta 0 --alpha 2` |
| **合并** | `python3 rydberg_chain/merge_vmc_csvs.py`<br>同上参数指定子目录 | `python3 long_range_ising/merge_vmc_csvs.py`<br>同上参数指定子目录 |
| **单精度图** | `python3 rydberg_chain/Fig_Convergence_Obs.py -p complex64`<br>可选 `--L --Rb --delta --alpha` | `python3 long_range_ising/Fig_Convergence_Obs.py -p complex64`<br>可选 `--L --J --delta --alpha` |
| **双精度对比图** | `python3 rydberg_chain/Fig_Convergence_Obs_compare.py`<br>可选 `--L --Rb --delta --alpha` | `python3 long_range_ising/Fig_Convergence_Obs_compare.py`<br>可选 `--L --J --delta --alpha` |

- 解析：默认从 **`train/<precision>/<param_subdir>/`** 下找对应 `.log`，写出 **`*_runN_parsed.csv`**、**`*_runN_summary.csv`**（N 可 `-r` 指定）。long_range_ising 的 parsed CSV 含 **sigma_Mx、sigma_Mz、sigma_Mz_AFM**。
- 合并：在同一子目录下合并为 **`*_merged_parsed.csv`**、**`*_merged_summary.csv`**（含 `run`、`global_iter`）。
- 图保存到各模块 **`figure/`**；long_range_ising 图含 \|Mx\|、\|Mz\|、\|Mz_AFM\| 及可选误差带。

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
