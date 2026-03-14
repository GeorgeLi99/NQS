import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import matplotlib.pyplot as plt

def compute_ground_state(L, J, alpha, h, delta):
    """
    Compute the ground state energy and order parameters for the long-range Ising model.

    Parameters:
        L (int): Chain length
        J (float): Coupling strength
        alpha (float): Long-range exponent
        h (float): Transverse field strength
        delta (float): Longitudinal field strength 

    Returns:
        tuple: (ground_state_energy, afm_z_correlation, transverse_x_magnetization)
    """
    # Create spin basis for 1D chain (full space, no symmetry restrictions)
    basis = spin_basis_1d(L, S="1/2")

    # Define long-range Ising interactions (z-z terms) with PBC
    Jz_list = []
    for i in range(L):
        for j in range(i + 1, L):
            r = min(abs(i - j), L - abs(i - j))
            factor = J / r**alpha
            Jz_list.append([factor, i, j])
    static_zz = [["zz", Jz_list]]

    # Define transverse field (x terms)
    hx_list = [[h, i] for i in range(L)]
    delta_list = [[-delta,i] for i in range(L)]
    static_x = [["x", hx_list]]
    static_z = [["z", delta_list]]

    # Construct Hamiltonian
    H = hamiltonian(static_zz + static_x + static_z, [], basis=basis, dtype=np.float64)

    # Use Lanczos method to find ground state (smallest eigenvalue)
    E, V = H.eigsh(k=1, which='SA')
    gs_energy = E[0]
    gs = V[:, 0]  # Ground state vector

    ############### Compute order parameters

    # zz-correlation function: $$C^{zz}(i,j) = \langle \sigma^z_i \sigma^z_j \rangle$$
    L_half = int(L/2)
    czz_ops = [hamiltonian([["zz", [[1, 0, j]]]], [], basis=basis) for j in range(L_half)]
    czz_values = [czz_ops[i].expt_value(gs).real for i in range(L_half)]
    czz_vec = np.array(czz_values)

    # Transverse x magnetization: sum_i <σ_i^x> / L
    Sx_ops = [hamiltonian([["x", [[1, i]]]], [], basis=basis) for i in range(L)]
    mx_values = [Sx_ops[i].expt_value(gs).real for i in range(L)]
    transverse_x_magnetization = sum(mx_values) / L

    return gs_energy, czz_vec, transverse_x_magnetization

# Parameters for alpha scan with fixed J
L = 16  # Chain length
hx = 1.0  # Fixed transverse field
J_fixed = 2.0  # Fixed J
delta = 0.0
alpha_list = [0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ]

# Initialize arrays to store results
energy_list = []
afm_z_list = []
transverse_x_list = []

# Scan over alpha with fixed J
for alpha in alpha_list:
    print(f"Computing for J={J_fixed}, alpha={alpha}")
    gs_energy, czz, transverse_x = compute_ground_state(L, J_fixed, alpha, hx, delta)
    # (a) GS energy 
    energy_list.append(gs_energy)
    # (b) AFM magnetization
    # afm_z_list.append(np.min(np.abs(czz[4:-1])))
    afm_z_list.append( np.abs(czz[-1]) )
    # (c) transverse magnetization 
    transverse_x_list.append(transverse_x)

# Convert to numpy arrays
energy_list = np.array(energy_list)
afm_z_list = np.array(afm_z_list)
transverse_x_list = np.array(transverse_x_list)

# Save data to txt files
np.savetxt(f'data/L={L}_delta={delta}_J={J_fixed}_alpha_vals.txt', np.array(alpha_list))
np.savetxt(f'data/L={L}_delta={delta}_J={J_fixed}_energy.txt', np.array(energy_list))
np.savetxt(f'data/L={L}_delta={delta}_J={J_fixed}_AFM.txt', np.array(afm_z_list))
np.savetxt(f'data/L={L}_delta={delta}_J={J_fixed}_Mx.txt', np.array(transverse_x_list))


# plot the figure
import matplotlib.pyplot as plt
# LaTex text 
import matplotlib as mpl
# set up of plot 
mpl.rcParams['text.usetex'] = True # the default is false, we set True here
mpl.rcParams['lines.linewidth'] = 1
plt.rcParams['font.family'] = ['Times New Roman']

# figure size
plt.figure(figsize=(8.6, 6.45))

# Two set of colors ... 
# (a) 
color_plan = ["#f58b47", "#fcce25",
          "#6300a7", "#a51f99",
          "#b7ea63",]
# (b)
# cmap = plt.get_cmap('viridis')
# num_colors = 8
# colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]


# (1) Order parameters  
ax1=plt.subplot(111)
# set up of plot 
mpl.rcParams['text.usetex'] = True # the default is false, we set True here
mpl.rcParams['lines.linewidth'] = 1
plt.rcParams['font.family'] = ['Times New Roman']

# AFM z magnetization vs alpha
ax1.plot(alpha_list, np.abs(np.array(afm_z_list)), color=color_plan[0], 
    linewidth = 2.0, marker='o', markersize=10.0, markerfacecolor="None", 
    label=r'$C^{zz}(0,L/2)$',)
ax1.plot(alpha_list, np.abs(np.array(transverse_x_list)), color=color_plan[2], 
    linewidth = 2.0, marker='s', markersize=10.0, markerfacecolor="None", 
    label=r'$M_x$',)
ax1.set_title('Exact diagonalization, '+r'$J=$'+f'{J_fixed}, '+r'$L=$'+f"{L}", 
              fontsize = 25.0)
ax1.set_xlabel(r'$\alpha$', fontsize=25.0)
ax1.set_ylabel(r"$C^{zz}(0,L/2)$"+" or "+r"$M_x$", fontsize = 25.0)

ax1.set_xlim((0.0,8.0))
ax1.set_ylim((0.0,1.0))

ax1.set_xticks([0, 2, 4, 6, 8],
    ['0','2','4','6', '8'], fontsize=25,)

ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8,1.0 ],
    ['0.0','0.2','0.4','0.6','0.8','1.0'],
    fontsize=25,)

ax1.legend(loc='upper right',fontsize=25, frameon=False,)


import matplotlib.ticker as ticker
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

ax1.tick_params("both", which='major', length=4, # width=1.0, 
    direction='in',#labelsize=12.5
    )
ax1.tick_params("both", which='minor', length=2, # width=1.0, 
    direction='in',#labelsize=12.5
    )

# axes[1].grid(True)

plt.tight_layout()
plt.savefig(f'figs/Ords_L={L}_delta={delta}_J={J_fixed}.pdf')
plt.savefig(f'figs/Ords_L={L}_delta={delta}_J={J_fixed}.svg')
plt.show()