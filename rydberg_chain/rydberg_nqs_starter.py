#!/usr/bin/env python3
"""
================================================================================
Neural Quantum States for Rydberg Atom Arrays - Starter Code
================================================================================

This script demonstrates how to use Neural Quantum States (NQS) to find the
ground state of a 1D Rydberg atom chain using Variational Monte Carlo (VMC).

PHYSICAL SYSTEM:
The Hamiltonian describes Rydberg atoms in a 1D chain with:
  H = (Ω/2) Σᵢ σˣᵢ - δ Σᵢ nᵢ + Σᵢ<ⱼ (C₆/rᵢⱼ^α) nᵢ nⱼ

where:
  - σˣᵢ: Pauli X operator (coherent driving between ground and Rydberg states)
  - nᵢ = (1 + σᶻᵢ)/2: occupation operator (1 if excited, 0 if ground)
  - Ω: Rabi frequency (transverse field strength)
  - δ: detuning (chemical potential for excitations)
  - C₆/rᵢⱼ^α: long-range van der Waals interaction
  - α: interaction exponent (α=6 for Rydberg atoms)

In spin-1/2 language (|↑⟩ = excited, |↓⟩ = ground):
  H = (Ω/2) Σᵢ σˣᵢ - δ̃ Σᵢ σᶻᵢ + Σᵢ<ⱼ (J/rᵢⱼ^α) σᶻᵢ σᶻⱼ

This is a transverse-field Ising model with long-range interactions.

PROJECT TASKS (see notes/NQS project.md):
  1. Exact diagonalization benchmarking (Task 3.1)
  2. NQS vs DMRG comparison (Task 3.2)
  3. Varying interaction exponent α (Task 3.3) ← CENTRAL TASK
  4. Phase diagram mapping (Task 3.4)

USAGE:
  python3 rydberg_nqs_starter.py

OUTPUT:
  - Log file with energy, variance, and observables per iteration
  - Can be analyzed with NetKet's plotting tools

AUTHOR: Zhiling Wei
DATE: 2026
================================================================================
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax as opx
import flax.serialization
from flax import linen as nn
from netket.operator.spin import sigmax, sigmaz

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

# Check JAX version and GPU availability
print(f"JAX version: {jax.__version__}") # 应显示 0.4.38
print(f"Is GPU available: {jax.devices()[0].device_kind}")

# Set computation backend (CPU or GPU)
# Change to "gpu" if you have CUDA installed
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# For reproducibility: set random seed
# TODO (Task 3.1+): Uncomment and use consistent seeds for all calculations
# RANDOM_SEED = 42
# jax.random.PRNGKey(RANDOM_SEED)

# ============================================================================
# ADVANCED: MPI Parallel Computing (Optional)
# ============================================================================
# For large-scale calculations, you can distribute sampling across multiple
# CPU cores or compute nodes using MPI (Message Passing Interface).
#
# WHEN TO USE:
# - Large system sizes (L > 20)
# - Need many samples for good statistics
# - Have access to multi-core or cluster computing
# - Want to speed up calculations
#
# HOW TO IMPLEMENT:
# 1. Add MPI imports at the top:
#    from mpi4py import MPI
#    comm = MPI.COMM_WORLD
#    rank = comm.Get_rank()
#
# 2. Run with mpirun:
#    mpirun -np 4 python3 rydberg_nqs_starter.py
#    (This uses 4 processes)
#
# 3. NetKet automatically distributes sampling across MPI ranks
#    No code changes needed beyond imports!
#
# BENEFITS:
# - Linear speedup with number of processes (for sampling)
# - Can handle larger systems
# - Better statistics in same wall-clock time
#
# NOTE: The example code includes MPI setup. See rbm_rydberg_v1.py
# lines 23-25 for example.
# ============================================================================

print("=" * 80)
print("Neural Quantum States for Rydberg Atoms")
print("=" * 80)
print(f"NetKet version: {nk.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 80)

start_time = time.time()

# ============================================================================
# SECTION 2: PHYSICAL PARAMETERS
# ============================================================================
# These parameters define the Hamiltonian and system size.
# Students should modify these to explore different regimes.

# --- Computation precision (complex64 or complex128 only) ---
PRECISION = "complex64"  # "complex64" (faster, less memory) or "complex128" (higher precision)
if PRECISION not in ("complex64", "complex128"):
    raise ValueError(f"PRECISION must be 'complex64' or 'complex128', got: {PRECISION!r}")
dtype_np = np.complex64 if PRECISION == "complex64" else np.complex128
dtype_jnp = jnp.complex64 if PRECISION == "complex64" else jnp.complex128
print(f"\nComputation precision: {PRECISION} (numpy: {dtype_np}, JAX: {dtype_jnp})")

# System size
L = 16  # Number of sites in the chain
print(f"\nSystem size: L = {L}")

# Physical parameters (in units where ℏ = 1)
Omega = 1.0      # Rabi frequency (transverse field strength)
delta = 0.5      # Detuning (longitudinal field)
Rb = 1.0         # Blockade radius (sets interaction strength J = Rb^6)

print(f"Physical parameters:")
print(f"  Ω (Rabi frequency) = {Omega}")
print(f"  δ (detuning) = {delta}")
print(f"  Rb (blockade radius) = {Rb}")

# Interaction parameters
C6 = Rb**6  # Van der Waals coefficient
# TODO (Task 3.3): Make this a variable parameter!
# This is THE KEY MODIFICATION for the project.
# Currently hardcoded to α=6 (van der Waals).
# Students should vary α ∈ {1, 1.5, 2, 3, 6} to test when NQS beats DMRG.
alpha_interaction = 6  # Interaction exponent

print(f"  C₆ = {C6}")
print(f"  α (interaction exponent) = {alpha_interaction}")
print(f"\nNote: α > 2 → area law (DMRG efficient)")
print(f"      α ≤ 2 → area law breaks down (NQS advantage expected)")

# ============================================================================
# SECTION 3: HILBERT SPACE AND GRAPH
# ============================================================================
# Define the quantum system: 1D chain of spin-1/2 particles

# Graph: 1D chain with periodic boundary conditions (PBC)
graph = nk.graph.Chain(length=L, pbc=True)
print(f"\nGraph: 1D chain with {L} sites, periodic boundary conditions")

# Hilbert space: spin-1/2 at each site
# |↑⟩ represents Rydberg excited state |e⟩
# |↓⟩ represents ground state |g⟩
hilbert = nk.hilbert.Spin(s=0.5, N=L)
print(f"Hilbert space dimension: 2^{L} = {2**L}")

# ============================================================================
# SECTION 4: HAMILTONIAN CONSTRUCTION
# ============================================================================
# Build the Rydberg Hamiltonian term by term

print(f"\nConstructing Hamiltonian...")

# Matrix representations of operators (use selected precision)
# Occupation operator: n = |e⟩⟨e| = (1 + σᶻ)/2
N_matrix = np.array([[1, 0],
                     [0, 0]], dtype=dtype_np)

# Pauli X: σˣ = |e⟩⟨g| + |g⟩⟨e|
Sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=dtype_np)

# Initialize Hamiltonian
H = nk.operator.LocalOperator(hilbert, dtype=dtype_np)

# Term 1: Transverse field (Ω/2) Σᵢ σˣᵢ
# This drives coherent transitions between |g⟩ and |e⟩
print(f"  Adding transverse field term: (Ω/2) Σᵢ σˣᵢ")
for i in range(L):
    H += (Omega/2) * nk.operator.LocalOperator(hilbert, Sigma_x, [i])

# Term 2: Detuning -δ Σᵢ nᵢ
# This acts as a chemical potential for excitations
print(f"  Adding detuning term: -δ Σᵢ nᵢ")
for i in range(L):
    H += -delta * nk.operator.LocalOperator(hilbert, N_matrix, [i])

# Term 3: Long-range interactions Σᵢ<ⱼ (C₆/rᵢⱼ^α) nᵢ nⱼ
# This is the Rydberg blockade: excited atoms repel each other
print(f"  Adding long-range interactions: Σᵢ<ⱼ (C₆/r^α) nᵢ nⱼ")
print(f"    (This may take a moment for large L...)")

interaction_count = 0
for i in range(L):
    for j in range(i+1, L):  # Sum over i < j to avoid double counting
        # Distance with periodic boundary conditions
        # Take the shorter of the two paths around the ring
        r = min(abs(i - j), L - abs(i - j))

        # Interaction strength: C₆/r^α
        # TODO (Task 3.3): Replace the hardcoded exponent 6 with alpha_interaction
        # This is where you modify the code to test different interaction ranges!
        interaction_strength = C6 / r**alpha_interaction

        # Add nᵢ nⱼ term
        H += interaction_strength * (
            nk.operator.LocalOperator(hilbert, N_matrix, [i]) @
            nk.operator.LocalOperator(hilbert, N_matrix, [j])
        )
        interaction_count += 1

print(f"  Added {interaction_count} interaction terms")
print(f"Hamiltonian construction complete!")

# ============================================================================
# SECTION 5: NEURAL NETWORK ANSATZ (WAVEFUNCTION)
# ============================================================================
# Define the neural network that represents the quantum state ψ(σ)
# We use a Restricted Boltzmann Machine (RBM) as a simple baseline

print(f"\nDefining neural network ansatz...")

# RBM hyperparameter: α controls the number of hidden units
# Number of hidden units = α × number of visible units (L)
rbm_alpha = 4  # Typical values: 1-8
print(f"  RBM with α = {rbm_alpha} (hidden units = {rbm_alpha * L})")

# Create RBM model
# The RBM represents log(ψ(σ)) as a neural network
model = nk.models.RBM(
    alpha=rbm_alpha,
    param_dtype=dtype_jnp,  # Use selected precision (complex64 or complex128)
    use_hidden_bias=True,
    kernel_init=nn.initializers.normal(stddev=0.01),
    hidden_bias_init=nn.initializers.normal(stddev=0.01),
)

print(f"  Model: Restricted Boltzmann Machine (RBM)")
print(f"  Parameter type: {PRECISION}")

# ============================================================================
# ADVANCED: Custom Neural Network Architectures (Optional)
# ============================================================================
# The RBM above is a simple baseline. For more complex ground states,
# custom architectures can provide better expressivity.
#
# WHEN TO USE:
# - RBM doesn't converge well
# - Working with strongly correlated systems
# - Need better accuracy for research
# - Exploring state-of-the-art NQS methods
#
# AVAILABLE ARCHITECTURES:
#
# 1. DEEP MLP WITH AMPLITUDE-PHASE SEPARATION
#    - Separate networks for |ψ| and arg(ψ)
#    - Multiple hidden layers
#    - Parity projection for Z₂ symmetry
#    - See: class_mlp_structure_parity_dense_control.py
#
#    Example usage:
#    import sys
#    sys.path.append('')
#    import class_mlp_structure_parity_dense_control as cus_mlp
#
#    model = cus_mlp.AmpPhaseMLP(
#        alpha=4,
#        param_dtype=jnp.float32,
#        hidden_activation=jax.nn.gelu,
#        use_hidden_bias=True,
#        equal_amplitude=False  # Set True to only learn phase
#    )
#
# 2. CONVOLUTIONAL NETWORKS
#    - Good for systems with spatial structure
#    - Use nk.models.GCNN for graph convolutional networks
#
# 3. TRANSFORMER-BASED
#    - State-of-the-art for some systems
#    - Requires more careful tuning
#
# TRADE-OFFS:
# - More expressive → Better accuracy
# - More parameters → Slower, needs more samples
# - More complex → Harder to debug and tune
#
# RECOMMENDATION: Start with RBM, only use custom architectures if needed
# ============================================================================

# ============================================================================
# ADVANCED: Symmetry Projection (Optional)
# ============================================================================
# For better convergence, you can project the wavefunction onto a specific
# symmetry sector. This reduces the effective parameter space and enforces
# physical symmetries exactly.
#
# WHEN TO USE:
# - System has known symmetries (translation, inversion, etc.)
# - Want faster convergence
# - Working with larger systems
#
# HOW TO IMPLEMENT:
# 1. Define symmetry group (e.g., translation group)
# 2. Wrap model with nk.nn.blocks.SymmExpSum
# 3. Use wrapped model in MCState
#
# EXAMPLE CODE (see mlp2_rydberg_symm.py for full version):
#
# from netket.utils.group import PermutationGroup, Permutation
#
# def OneSiteTranslationGroup(L: int):
#     """Create translation symmetry group T(k) for k=0,...,L-1"""
#     group_elems = []
#     for k in range(L):
#         perm = [(i + k) % L for i in range(L)]
#         group_elems.append(Permutation(perm, name=f"T({k})"))
#     return PermutationGroup(elems=group_elems, degree=L)
#
# group = OneSiteTranslationGroup(L)
# model_symm = nk.nn.blocks.SymmExpSum(module=model, symm_group=group)
# # Then use model_symm instead of model in MCState below
#
# BENEFITS:
# - Faster convergence (fewer effective parameters)
# - Exact symmetry enforcement
# - Can target specific quantum numbers (momentum, parity)
#
# ============================================================================

# ============================================================================
# SECTION 6: VARIATIONAL MONTE CARLO SETUP
# ============================================================================
# Configure the VMC algorithm: sampler, optimizer, and preconditioner

print(f"\nConfiguring Variational Monte Carlo...")

# --- Monte Carlo Sampler ---
# Generates spin configurations σ sampled from |ψ(σ)|²
N_samples = 1024 * 32  # Total number of samples per iteration
n_chains_per_rank = 512  # Number of independent Markov chains
sweep_size = 4 * L  # MC steps between samples

sampler = nk.sampler.MetropolisLocal(
    hilbert=hilbert,
    n_chains_per_rank=n_chains_per_rank,
    sweep_size=sweep_size
)

print(f"  Sampler: Metropolis Local")
print(f"  Total samples per iteration: {N_samples}")
print(f"  Chains per rank: {n_chains_per_rank}")
print(f"  Sweep size: {sweep_size}")

# --- Optimizer ---
# Stochastic Gradient Descent (SGD) with constant learning rate
learning_rate = 0.004  # Typical values: 0.001 - 0.01
optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

print(f"  Optimizer: SGD")
print(f"  Learning rate: {learning_rate}")

# ============================================================================
# ADVANCED: Learning Rate Schedules (Optional)
# ============================================================================
# For difficult optimization problems, adaptive learning rate schedules can
# help avoid early divergence and improve final convergence.
#
# WHEN TO USE:
# - Optimization is unstable at the start
# - Want to fine-tune near the minimum
# - Working with complex systems or architectures
#
# HOW TO IMPLEMENT:
# Replace the constant learning rate above with a schedule:
#
# import optax as opx
# learning_rate = opx.warmup_cosine_decay_schedule(
#     init_value=0.001,      # Start low
#     peak_value=0.01,       # Ramp up to this
#     warmup_steps=50,       # Over this many steps
#     decay_steps=950,       # Then decay over remaining steps
#     end_value=0.001        # End at this value
# )
# optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
#
# BENEFITS:
# - Warmup prevents early instability
# - Cosine decay allows fine-tuning near minimum
# - Often better final convergence
#
# See rbm_rydberg_v1.py lines 166-174 for example
# ============================================================================

# --- Stochastic Reconfiguration (SR) ---
# Natural gradient method that accounts for the geometry of parameter space
# This significantly improves convergence compared to plain SGD
diag_shift = 1e-4  # Regularization parameter

preconditioner = nk.optimizer.SR(
    diag_shift=diag_shift,
    holomorphic=True  # Assume holomorphic wavefunction
)

print(f"  Preconditioner: Stochastic Reconfiguration (SR)")
print(f"  Diagonal shift: {diag_shift}")

# --- Variational State ---
# Combines the model, sampler, and sampling parameters
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_discard_per_chain=32,  # Thermalization: discard first N samples
    chunk_size=1024 * 8,  # Process samples in chunks (for memory efficiency)
    n_samples=N_samples
)

print(f"  Variational state created")
print(f"  Number of parameters: {vstate.n_parameters}")

# ============================================================================
# SECTION 7: OBSERVABLES
# ============================================================================
# Define physical quantities to measure during optimization

print(f"\nDefining observables...")

# Average magnetization in x-direction: ⟨Mˣ⟩ = (1/L) Σᵢ ⟨σˣᵢ⟩
Mx = nk.operator.LocalOperator(hilbert, dtype=dtype_np)
for i in range(L):
    Mx += (1/L) * sigmax(hilbert, i)

# Average magnetization in z-direction: ⟨Mᶻ⟩ = (1/L) Σᵢ ⟨σᶻᵢ⟩
# This is the order parameter for the Ising transition
Mz = nk.operator.LocalOperator(hilbert, dtype=dtype_np)
for i in range(L):
    Mz += (1/L) * sigmaz(hilbert, i)

# Average occupation: ⟨n⟩ = (1/L) Σᵢ ⟨nᵢ⟩
# Measures the density of Rydberg excitations
Ntot = nk.operator.LocalOperator(hilbert, dtype=dtype_np)
for i in range(L):
    Ntot += (1/L) * nk.operator.LocalOperator(hilbert, N_matrix, [i])

observables = {
    'Mx': Mx,
    'Mz': Mz,
    'Ntot': Ntot,
}

print(f"  Observables: Mx (transverse mag.), Mz (longitudinal mag.), Ntot (occupation)")

# ============================================================================
# ADVANCED: Local Observables (Optional)
# ============================================================================
# Instead of measuring averaged quantities, you can measure site-resolved
# observables to study spatial structure, correlations, or inhomogeneities.
#
# WHEN TO USE:
# - Studying spatial patterns (e.g., density waves)
# - Computing correlation functions
# - Investigating symmetry breaking
# - Analyzing finite-size effects
#
# HOW TO IMPLEMENT:
# Add site-resolved operators to the observables dictionary:
#
# observables = {
#     'Mx': Mx, 'Mz': Mz, 'Ntot': Ntot,  # Keep global observables
#     **{f'n{i}': nk.operator.LocalOperator(hilbert, N_matrix, [i])
#        for i in range(L)},  # Add local occupation at each site
#     **{f'sz{i}': sigmaz(hilbert, i)
#        for i in range(L)},  # Add local magnetization at each site
# }
#
# WARNING: This increases output file size significantly!
#
# ANALYSIS:
# With local observables, you can compute:
# - Density profiles: n(i) vs i
# - Correlation functions: ⟨n(i)n(j)⟩ - ⟨n(i)⟩⟨n(j)⟩
# - Structure factor: S(q) = Σᵢⱼ exp(iq(i-j)) ⟨n(i)n(j)⟩
# ============================================================================

# ============================================================================
# SECTION 8: RUN VARIATIONAL MONTE CARLO
# ============================================================================
# Optimize the neural network parameters to minimize ⟨ψ|H|ψ⟩

print(f"\n" + "=" * 80)
print("Starting VMC optimization...")
print("=" * 80)

# Number of optimization iterations
n_iterations = 1000  # Typical: 1000-5000 depending on convergence

# Output directory: save .log and .mpack under rydberg_chain/train/<precision>/
# Matches PRECISION: "complex64" or "complex128"
TRAIN_SUBDIR = PRECISION
_script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(_script_dir, "train", TRAIN_SUBDIR)
os.makedirs(train_dir, exist_ok=True)
output_file = os.path.join(train_dir, f"rydberg_L{L}_delta{delta}_Rb{Rb}_alpha{alpha_interaction}")

# Optional: load checkpoint to resume training. Set to .mpack path (under train/<precision>/) or None to train from scratch.
LOAD_CHECKPOINT = os.path.join(train_dir, f"rydberg_L{L}_delta{delta}_Rb{Rb}_alpha{alpha_interaction}.mpack")  # e.g. os.path.join(train_dir, "rydberg_L16_delta0.5_Rb1.0_alpha6.mpack")

if LOAD_CHECKPOINT is not None and os.path.isfile(LOAD_CHECKPOINT):
    with open(LOAD_CHECKPOINT, "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
    print(f"  Loaded parameters from: {LOAD_CHECKPOINT}")
    print(f"  Number of parameters: {vstate.n_parameters}")

print(f"  Iterations: {n_iterations}")
print(f"  Output file: {output_file}.log")
print(f"\nOptimizing... (this may take several minutes)")

# Create VMC driver
vmc = nk.VMC(
    hamiltonian=H,
    optimizer=optimizer,
    variational_state=vstate,
    preconditioner=preconditioner
)

# Run optimization
# The .log file will contain:
#   - Energy (mean and error)
#   - Energy variance
#   - Observable values
#   - Acceptance rate
vmc.run(
    out=output_file,
    n_iter=n_iterations,
    obs=observables
)

# ============================================================================
# SECTION 9: RESULTS AND NEXT STEPS
# ============================================================================

elapsed_time = time.time() - start_time
print(f"\n" + "=" * 80)
print(f"VMC optimization complete!")
print(f"Total time: {elapsed_time:.2f} seconds")
print(f"=" * 80)

print(f"\nResults saved to: {output_file}.log")
print(f"\nTo analyze results:")
print(f"  1. Load the log file with NetKet's data utilities")
print(f"  2. Plot energy vs iteration to check convergence")
print(f"  3. Check that energy variance decreases")
print(f"  4. Examine observable values")

print(f"\n" + "=" * 80)
print("PROJECT TASKS - What to do next:")
print("=" * 80)
print(f"""
TASK 3.1: Exact Diagonalization Benchmark
  - For L ≤ 16, compute exact ground state energy
  - Compare with NQS result to verify accuracy
  - Code hint: use netket.exact.lanczos_ed(H, k=1)

TASK 3.2: NQS vs DMRG Comparison
  - Implement DMRG using TeNPy or ITensor
  - Compare convergence speed and accuracy
  - Track: bond dimension (DMRG) vs variance (NQS)

TASK 3.3: Vary Interaction Exponent α ← CENTRAL TASK
  - Modify line ~124: change r**6 to r**alpha_interaction
  - Run for α ∈ {{1, 1.5, 2, 3, 6}}
  - Compare NQS vs DMRG performance for each α
  - Identify where NQS has advantage (likely α < 2)

TASK 3.4: Phase Diagram
  - Once you identify optimal α for NQS, map (δ, Rb) plane
  - Compute order parameter (e.g., |⟨Mᶻ⟩|) on a grid
  - Identify phase boundaries
  - Discuss physics: disordered vs ordered phases

See notes/NQS project.md for detailed instructions.
""")

print("=" * 80)
