import os
import csv
import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

# Parameters
L = 16     # Chain length (adjust as needed for computational feasibility)
J = 1.0    # Coupling strength (long-range zz)
alpha = 2.0  # Long-range exponent
delta = 0.5  # Longitudinal field strength (couples to sigma^z)
h = 1.0    # Transverse field strength (couples to sigma^x)

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

# Define transverse field (x terms):  (Omega/2) * sum_i sigma^x_i  ~  h * sum_i sigma^x_i
hx_list = [[h, i] for i in range(L)]
static_x = [["x", hx_list]]

# Define longitudinal field (z terms):  -delta * sum_i sigma^z_i
hz_list = [[-delta, i] for i in range(L)]
static_z = [["z", hz_list]]

# Construct Hamiltonian
H = hamiltonian(static_zz + static_x + static_z, [], basis=basis, dtype=np.float64)

# Use Lanczos method to find ground state (smallest eigenvalue)
E, V = H.eigsh(k=1, which='SA')
gs_energy = E[0]
gs = V[:, 0]  # Ground state vector

print(f"Ground state energy: {gs_energy}")

# Compute order parameters

# AFM z magnetization: sum_i (-1)^i <σ_i^z> / L
Sz_ops = [hamiltonian([["z", [[1, i]]]], [], basis=basis) for i in range(L)]
mz_values = [Sz_ops[i].expt_value(gs).real for i in range(L)]
afm_z_magnetization = sum((-1)**i * mz_values[i] for i in range(L)) / L

# Transverse x magnetization: sum_i <σ_i^x> / L
Sx_ops = [hamiltonian([["x", [[1, i]]]], [], basis=basis) for i in range(L)]
mx_values = [Sx_ops[i].expt_value(gs).real for i in range(L)]
transverse_x_magnetization = sum(mx_values) / L

# Output results
print(f"Chain length L = {L}")
print(f"Coupling J = {J}, alpha = {alpha}, transverse field h = {h}")
print(f"Ground state energy: {gs_energy}")
print(f"AFM z magnetization: {afm_z_magnetization}")
print(f"Transverse x magnetization: {transverse_x_magnetization}")

# Save results to CSV in ED/result, named by parameters
base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, "result")
os.makedirs(result_dir, exist_ok=True)

csv_name = f"ising_L{L}_J{J}_alpha{alpha}_delta{delta}_h{h}.csv"
csv_path = os.path.join(result_dir, csv_name)

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["L", "J", "alpha", "delta", "h", "gs_energy", "afm_z_magnetization", "transverse_x_magnetization"])
    writer.writerow([L, J, alpha, delta, h, gs_energy, afm_z_magnetization, transverse_x_magnetization])

print(f"Results saved to CSV: {csv_path}")