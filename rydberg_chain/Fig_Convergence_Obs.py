import numpy as np
import json

def load_log(key_log, E0):
    data=json.load(open(f"{key_log}"))
    
    # iteration information 
    iters = data['Energy']['iters']
    energy=data['Energy']['Mean']['real']
    
    print("keys:", data.keys())
    
    # relative error
    error = np.array(energy)-E0 
    relative_error = np.abs(error/E0)

    return np.array(iters), np.array(energy), error, relative_error

def ConvergenceData(key_list,E0):
    """ Load a list of log, merge them into a vector."""
    N_log = len(key_list)
    ites = np.array([], dtype=float)
    rela_error = np.array([], dtype=float)
    print("shape of init:",ites.shape)
    
    for k in range(N_log):
        ite_k, energy_k, error_k, rela_error_k = load_log(key_list[k], E0)
        print("shape of load:",ite_k.shape)
        ites = np.concatenate( (ites, ite_k + ites.shape[0]+1), axis=0)
        rela_error = np.concatenate((rela_error, rela_error_k), axis=0)
        
    return ites,rela_error

def load_obs(key_log,):
    data=json.load(open(f"{key_log}"))
    
    # iteration information 
    iters = data['Mx']['iters']
    ntot =data['Ntot']['Mean']['real']
    mz =data['Mz']['Mean']['real']
    print("keys:", data.keys())

    return np.array(iters), np.array(ntot), np.array(mz)

def ObservablesData(key_list,):
    """ Load a list of log, merge them into a vector."""
    N_log = len(key_list)
    ites = np.array([], dtype=float)
    ntot = np.array([], dtype=float)
    mz = np.array([], dtype=float)
    print("shape of init:",ites.shape)
    
    for k in range(N_log):
        ite_k, ntot_k, mz_k = load_obs(key_list[k])
        print("shape of load:",ite_k.shape)
        ites = np.concatenate( (ites, ite_k + ites.shape[0]+1), axis=0)
        ntot = np.concatenate((ntot, ntot_k), axis=0)
        mz = np.concatenate((mz, mz_k), axis=0) 
    return ites,ntot,mz

### main code
L = 16

E0 = -8.878144715543531

plot_key = "Energy_and_Obs"

alpha = 6

# 与 rydberg_nqs_starter.py 输出路径一致：train/<precision>/
cal1 =  [
    "train/complex128/rydberg_L16_delta0.5_Rb1.0_alpha6.log",
]

print("GS energy:",E0)
data1 = ConvergenceData(cal1, E0)
dataObs1 = ObservablesData(cal1, )


colors = ["#085293", "#90d4bd",
    "#f58b47", "#fcce25",
    "#6300a7", "#a51f99",
    "#b7ea63",]

# plot the figure
import matplotlib.pyplot as plt
# LaTex text 
import matplotlib as mpl
# set up of plot 
mpl.rcParams['text.usetex'] = True # the default is false, we set True here
mpl.rcParams['lines.linewidth'] = 1
plt.rcParams['font.family'] = ['Times New Roman']

# figure size
plt.figure(figsize=(2*8.6, 6.45))

# (1) relative error
ax1=plt.subplot(121)

ax1.plot(data1[0], data1[1], color=colors[0], lw=2.0, 
         label=r"$\delta=0.5, \alpha = $"+f" {alpha}",)

ax1.set_ylabel(r'$\epsilon = |\frac{E-E_0}{E_0}|$',
               fontsize=25)
ax1.set_xlabel('Iteration',fontsize=25)

ax1.set_xlim((0,2000))
ax1.set_ylim((1E-6,1))
ax1.set_yscale('log')

ax1.set_xticks([0, 500, 1000, 1500, 2000],
    ['0',r'500',r'1000',r'1500 ', r'2000'],
    fontsize=25,)

ax1.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    [r'$10^0$',r'$10^{-1}$',r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$',r'$10^{-5}$',],
    fontsize=25,)


ax1.legend(loc='upper right',fontsize=25, frameon=False,)

ax1.tick_params("both", which='major', length=4, # width=1.0, 
    direction='in',#labelsize=12.5
    )
ax1.tick_params("both", which='minor', length=2, # width=1.0, 
    direction='in',#labelsize=12.5
    )

# (2) observables
ax2=plt.subplot(122)

ax2.plot(data1[0], np.abs(dataObs1[1]), color=colors[3], lw=2.0, 
         label=r"$D=0.0, |N_{tot}|, \alpha=$"+f" {alpha}", )
ax2.plot(data1[0], np.abs(dataObs1[2]), color=colors[4], lw=2.0, 
         label=r"$D=0.0, |M_z|, \alpha=$"+f" {alpha}", )

ax2.set_ylabel(r'$M_z = \frac{1}{L}\sum_{j=1}^L \langle \sigma^z_j\rangle$',
               fontsize=25)
ax2.set_xlabel('Iteration',fontsize=25)

ax2.set_xlim((0,2000))
ax2.set_ylim((1e-5,1.0))

# ax1.set_yscale('log')

ax2.set_xticks([0, 500, 1000, 1500, 2000],
    ['0',r'500',r'1000',r'1500', r'2000'],
    fontsize=25,)

ax2.legend(loc='upper right',fontsize=25, frameon=False,)

ax2.tick_params("both", which='major', length=4, # width=1.0, 
    direction='in',#labelsize=12.5
    )
ax2.tick_params("both", which='minor', length=2, # width=1.0, 
    direction='in',#labelsize=12.5
    )

# Big title
plt.suptitle(r"$\delta = 0.5, L=$"+f"{L},"+" RBM, "+r"$\alpha=$"+f"{alpha}", fontsize=25)
# adjust the space
plt.subplots_adjust(hspace=0.25, wspace=0.25) 
plt.tight_layout()
plt.show()

plt.savefig(f"Fig_ConvObs_RBM_delta=0.5_L={L}_{plot_key}.pdf")
plt.savefig(f"Fig_ConvObs_RBM_delta=0.5_L={L}_{plot_key}.svg")