import sys
sys.path.append('../../') 
sys.path.append('../') 
# GPU support
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import netket as nk
import numpy as np
from netket import experimental as nkx
from jax import random
from netket import jax as nkjax

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax as opx
from flax import nnx

import scipy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import json
import msgpack
import time
import shutil
print("netket version: ", nk.__version__)
print(f"Using platform: {jax.default_backend()}")

start_time = time.time()
from jax import vmap

from netket.utils.group import PermutationGroup, Permutation

# S=1/2 long-range Ising model with transverse field
J = 1.0 
alpha_interaction = 2.0
Omega = 1.0
delta = 0.5
print(f"Ising J = {J}, alpha_interaction = {alpha_interaction}, Omega = {Omega}, delta = {delta}")

L = 16
Nc = L
print("number of cells:",Nc)
alpha_rbm = 4
print("Feature alpha=",alpha_rbm)
use_bias = False
print("use_bias=",use_bias)

key_cal = 1 # val_cal_times

# Graph
g = nk.graph.Chain(length=L, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5,  N=g.n_nodes, #total_sz= 0, 
    ) 

# Hamiltonian
from netket.operator.spin import sigmax, sigmay, sigmaz
Sigma_z = np.array([[1, 0], [0, -1],])
Sigma_x = np.array([[0, 1], [1, 0],])

H = nk.operator.LocalOperator(hi, dtype=complex)
# H = (Omega/2) * sum_j sigma^x_j
#     - delta * sum_j sigma^z_j
#     + sum_{i<j} (J / r_{ij}^{alpha_interaction}) * sigma^z_i sigma^z_j
# (1) trasverse field 
for j in range(L):
    H += (Omega / 2) * nk.operator.LocalOperator(hi, Sigma_x, [j])
# (1.5) longitudinal field
for j in range(L):
    H += (-delta) * nk.operator.LocalOperator(hi, Sigma_z, [j])
# (2) long-range Ising interaction
for c in range(L):
    # print(f"site_c: c = {c}")
    # summation for i < j
    for j in range(c+1, L):  
        #print(f"site_j: j = {j}")
        r = min(abs(c-j), L - abs(c-j))
        #print(f"Distance R = {r}")
        factor = J/r**alpha_interaction
        H += factor*nk.operator.LocalOperator(hi, Sigma_z, [c])@nk.operator.LocalOperator(hi, Sigma_z, [j])
op = H

# import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp

from netket.nn.activation import reim_selu
# from jax.nn.initializers import normal
model = nk.models.RBM(
    alpha=alpha_rbm, 
    param_dtype=jnp.complex128, 
    use_hidden_bias=True,        
    kernel_init=nn.initializers.normal(stddev=0.01), 
    hidden_bias_init=nn.initializers.normal(stddev=0.01),
)
print("RBM: alpha = ", alpha_rbm)

# VMC hyper-parameters
n_warmup = 50
diag_end_value = 1E-3
learning_end_value = 4E-3
learning_rate = opx.warmup_cosine_decay_schedule(init_value=1E-2, peak_value=4E-2, 
                                                 warmup_steps=n_warmup, decay_steps=(200-n_warmup), 
                                                 end_value=learning_end_value, exponent= 1.0 ) 
diag_shift = opx.warmup_cosine_decay_schedule(init_value=5E-2, peak_value=5E-2, 
                                                 warmup_steps=n_warmup, decay_steps=(200-n_warmup), 
                                                 end_value=diag_end_value, exponent= 1.0 )

# Optimizer 
# val_learning_rate = 0.001
val_learning_rate = 0.01
# val_diagonal_scale = 0.0001
val_diagonal_shfit = 0.0001 
print(f"constant learning rate: eta = {val_learning_rate}") 
# print(f"constant diagonal scale: epsilon_1 = {val_diagonal_scale}")
print(f"constant diagonal shift: epsilon_2 = {val_diagonal_shfit}")

# Stochastic Gradient Descent
opt_decay = nk.optimizer.Sgd(learning_rate=learning_rate)
opt_const = nk.optimizer.Sgd(learning_rate=val_learning_rate)
print("Constant optimizer:",opt_const)
print("Decay optimizer:",opt_decay)

# Stochastic Reconfiguration 
sr = nk.optimizer.SR(diag_shift=val_diagonal_shfit, holomorphic = True,)
print("SR preconditioner:",sr)

# MC sampler
N_samples = 1024*32
#sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, n_chains=N_samples, sweep_size= g.n_nodes, d_max=2,
#    reset_chains = True,)
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=512, sweep_size=4*L)
print("Sampler: ", sampler)

# VQS
vs = nk.vqs.MCState(sampler=sampler, model=model,
    n_discard_per_chain=32, chunk_size=1024*8, n_samples= N_samples,  )
print("Variational state:",vs)
print("Number of parameters:", vs.n_parameters)

# Variational Monte Carlo
gs = nk.VMC(hamiltonian=op, optimizer=opt_const, 
        # optimizer=opt_decay, 
        variational_state=vs, preconditioner=sr,)
print("VMC:",gs)

# observable
Mx = nk.operator.LocalOperator(hi, dtype=complex)
Mz = nk.operator.LocalOperator(hi, dtype=complex)
for j in range(0, L):
    Mx += (1/L)*sigmax(hi, j)
    Mz += (1/L)*sigmaz(hi, j)

local_obs = False
if local_obs == False: 
    obs={'Mx': Mx, 'Mz': Mz, }

n_iteration = 1000
# GS calculation
file_name = f"data/rbm_LongIsing_L={L}_J={J}_alphaInt={alpha_interaction}_alpha={alpha_rbm}_Cal{key_cal}"
print("Number of iterations:",n_iteration)
print("file name:",file_name)
gs.run(out=file_name, n_iter=n_iteration, obs=obs,)