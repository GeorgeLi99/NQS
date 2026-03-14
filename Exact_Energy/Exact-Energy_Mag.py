import numpy as np
import matplotlib.pyplot as plt

# 使用 SciencePlots 风格；不强制依赖系统 LaTeX（避免 'latex' not found 报错）
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

# Ref: PhysRevE101,042108(2020).


def single_k_mode(J: float, hx: float, n: int, L: int) -> float:
    return np.sqrt(J**2 + hx**2 + 2*J*hx*np.cos(2*np.pi*n/L))


def GS_energy(J: float, hx: float, L: int) -> float:
    l = -int(L/2)
    r = int(L/2)-1
    return -sum(single_k_mode(J, hx, n, L) for n in range(l, r+1))


def _single_mode_mag(J: float, hx: float, n: int, L: int) -> float:
    eps_n = single_k_mode(J, hx, n, L)
    return (J * np.cos(2*np.pi*n/L) + hx) / eps_n


def Mz_exact(J: float, hx: float, L: int) -> float:
    """
    M_z 精确解（来自 Ref. 论文中 Bogoliubov 公式，即横向磁化）。
    保持原有逻辑不变。
    """
    return sum(_single_mode_mag(J, hx, n, L) for n in range(L)) / L


def Mx_exact(J: float, hx: float) -> float:
    """
    Ising 序参量（纵向磁化）的热力学极限精确解析公式：
        M_x = [1 - (hx/J)^2]^{1/8},  hx < J
        M_x = 0,                       hx >= J
    对应论文中的 M_z（式65，Wu 2020, PhysRevE101,042108）。
    """
    if hx >= J:
        return 0.0
    return float((1.0 - (hx / J) ** 2) ** 0.125)


if __name__ == "__main__":
    import os

    L = 16
    J = 1.0
    hx_list_print = np.arange(0.0, 2.0, 0.1)
    for hx in hx_list_print:
        print(f"Energy = {GS_energy(J, hx, L):.6f}")

    # 解析公式（快速）
    hx_list = np.linspace(0.0, 2.0, 50)
    energies = [GS_energy(J, hx, L) for hx in hx_list]
    mz_vals  = [Mz_exact(J, hx, L) for hx in hx_list]

    mx_vals = [Mx_exact(J, hx) for hx in hx_list]

    # 三子图：E0 / M_z / M_x
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(hx_list, energies, 'o-', label=f'$L={L}$')
    axes[0].set_xlabel(r'$h_x$')
    axes[0].set_ylabel(r'$E_0$')
    axes[0].set_title('Ground State Energy')
    axes[0].legend()

    axes[1].plot(hx_list, mz_vals, 's-', color='steelblue', label=f'$L={L}$')
    axes[1].set_xlabel(r'$h_x$')
    axes[1].set_ylabel(r'$M_z$')
    axes[1].set_title('Longitudinal Magnetization $M_z$')
    axes[1].legend()

    axes[2].plot(hx_list, mx_vals, '^-', color='orange', label=f'$L={L}$')
    axes[2].set_xlabel(r'$h_x$')
    axes[2].set_ylabel(r'$M_x$')
    axes[2].set_title('Transverse Magnetization $M_x$ (Onsager)')
    axes[2].legend()

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    out_name = f"Exact-Energy_Mag_L{L}_J{J}_hx0.0-2.0.pdf"
    out_path = os.path.join(fig_dir, out_name)
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")
    plt.close()
