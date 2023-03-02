# Import the relevant libraries
import os
import pickle
import numpy as np

from qiskit.quantum_info import Statevector


import numpy             as     np
import matplotlib        as     mpl
import matplotlib.pyplot as     plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from   matplotlib        import rc
from   cycler            import cycler

import json

import time
import math
import pickle
import sys

from cmcrameri import cm

cmp = cm.vik.colors

_widths = {
    # a4paper columnwidth = 426.79135 pt = 5.93 in
    # letterpaper columnwidth = 443.57848 pt = 6.16 in
    'onecolumn': {
        'a4paper' : 5.93,
        'letterpaper' : 6.16
    },
    # a4paper columnwidth = 231.84843 pt = 3.22 in
    # letterpaper columnwidth = 240.24199 pt = 3.34 in
    'twocolumn': {
        'a4paper' : 3.22,
        'letterpaper' : 3.34
    }
}

_wide_widths = {
    # a4paper wide columnwidth = 426.79135 pt = 5.93 in
    # letterpaper wide columnwidth = 443.57848 pt = 6.16 in
    'onecolumn': {
        'a4paper' : 5.93,
        'letterpaper' : 6.16
    },
    # a4paper wide linewidth = 483.69687 pt = 6.72 in
    # letterpaper wide linewidth = 500.48400 pt = 6.95 in
    'twocolumn': {
        'a4paper' : 6.72,
        'letterpaper' : 6.95
    }
}

_fontsizes = {
    10 : {
        'tiny' : 5,
        'scriptsize' : 7,
        'footnotesize' : 8, 
        'small' : 9, 
        'normalsize' : 10,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' : 20,
        'Huge' : 25
    },
    11 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 9, 
        'small' : 10, 
        'normalsize' : 11,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' :  20,
        'Huge' :  25
    },
    12 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 10, 
        'small' : 11, 
        'normalsize' : 12,
        'large' : 14, 
        'Large' : 17, 
        'LARGE' : 20,
        'huge' :  25,
        'Huge' :  25
    }
}

_width         = 1
_wide_width    = 1
_quantumviolet = '#53257F'
_quantumgray   = '#555555'

# Sets up the plot with the fitting arguments so that the font sizes of the plot
# and the font sizes of the document are well aligned
#
#     columns : string = ('onecolumn' | 'twocolumn')
#         the columns you used to set up your quantumarticle, 
#         defaults to 'twocolumn'
#
#     paper : string = ('a4paper' | 'letterpaper')
#         the paper size you used to set up your quantumarticle,
#         defaults to 'a4paper'
#
#     fontsize : int = (10 | 11 | 12)
#         the fontsize you used to set up your quantumarticle as int
#
#     (returns) : dict
#         parameters that can be used for plot adjustments

def global_setup(columns = 'twocolumn', paper = 'a4paper', fontsize = 10):
    plt.rcdefaults()
        
    # Seaborn white is a good base style
    # print(plt.style.available)
    plt.style.use(['seaborn-v0_8-white', 'quantum-plots.mplstyle'])
    
    #try:        
        # This hackery is necessary so that jupyther shows the plots
    #    mpl.use("pgf")
        #%matplotlib inline
    #    plt.plot()
    #    mpl.use("pgf")
    #except:
    #    print('Call to matplotlib.use had no effect')
        
    #mpl.interactive(False) 
    
    # Now prepare the styling that depends on the settings of the document
    
    global _width 
    _width = _widths[columns][paper]
    
    global _wide_width 
    _wide_width = _wide_widths[columns][paper]
    
    # Use the default fontsize scaling of LaTeX
    global _fontsizes
    fontsizes = _fontsizes[fontsize]
    
    plt.rcParams['axes.labelsize']  = fontsizes['normalsize']
    plt.rcParams['axes.titlesize']  = fontsizes['normalsize']
    plt.rcParams['xtick.labelsize'] = fontsizes['footnotesize']
    plt.rcParams['ytick.labelsize'] = fontsizes['footnotesize']
    plt.rcParams['font.size']       = fontsizes['small']
    plt.rcParams['legend.fontsize'] = fontsizes['footnotesize']
    
    return {
            'fontsizes' : fontsizes,
            'colors' : {
                'quantumviolet' : _quantumviolet,
                'quantumgray' : _quantumgray
            }
        }
    

# Sets up the plot with the fitting arguments so that the font sizes of the plot
# and the font sizes of the document are well aligned
#
#     aspect_ratio : float
#         the aspect ratio (width/height) of your plot
#         defaults to the golden ratio
#
#     width_ratio : float in [0, 1]
#         the width of your plot when you insert it into the document, e.g.
#         .8 of the regular width
#         defaults to 1.0
#
#     wide : bool 
#         indicates if the figures spans two columns in twocolumn mode, i.e.
#         when the figure* environment is used, has no effect in onecolumn mode 
#         defaults to False
#
#     (returns) : matplotlib figure object
#         the initialized figure object

def plot_setup(aspect_ratio = 1/1.62, width_ratio = 1.0, wide = False):
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
           
    return plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    
print('Setup methods loaded')

props = global_setup(columns = 'twocolumn', paper = 'a4paper', fontsize = 11)

##########################################################################################################
'''
# Exact datasets
exact_datasets = {}
exact_dir = 'qutip_exact_states/'
for filename in os.listdir(exact_dir):
    N = filename.split('_')[4].split('=')[1]
    exact_datasets[N] = np.load(exact_dir+filename)

# Trotter fixed depth datasets
trotter_fixed_depth_datasets = {}
trot_n_dir = 'trotter_fixed_depth/'
for filename in os.listdir(trot_n_dir):
    N = filename.split('_')[5].split('=')[1]
    trotter_fixed_depth_datasets[N] = np.load(trot_n_dir+filename, allow_pickle=True)

# Trotter fixed dt datasets
trotter_fixed_dt_datasets = {}
trot_dt_dir = 'trotter_fixed_dt/'
for filename in os.listdir(trot_dt_dir):
    N = filename.split('_')[5].split('=')[1]
    trotter_fixed_dt_datasets[N] = np.load(trot_dt_dir+filename, allow_pickle=True)

# pVQD datasets
pvqd_datasets = {}
pvqd_dir = 'pvqd/'
for filename in os.listdir(pvqd_dir):
    N = filename.split('_')[5].split('=')[1]
    with open(pvqd_dir+filename, 'rb') as f:
        pvqd_datasets[N] = pickle.load(f)

# Adaptive pVQD, local pool, datasets
pool_datasets = {}
pool_dir = 'pool_adaptive/'
for filename in os.listdir(pool_dir):
    N = filename.split('_')[5].split('=')[1]
    with open(pool_dir+filename, 'rb') as f:
        pool_datasets[N] = pickle.load(f)


def exact_fidelity(exact_wvfs, approx_wvfs):
    """ Overlap between the exact time-evolved state and an approximation of it. """
    return np.abs(np.einsum('ij,ij->i', np.conj(exact_wvfs), approx_wvfs))**2

def circs_to_vectors(circ_list):
    """ Convert a list of Qiskit QuantumCircuits to an array of the corresponding statevectors. """
    return np.array([Statevector.from_instruction(qc) for qc in circ_list])

def count_cnots(circ_list, i):
    """ Count the # of CNOTs in a circuit at position i in a given list of circuits. """
    return circ_list[i].decompose(reps=10).count_ops()["cx"]

def measure_depth(circ_list, i):
    """ Measure the depth of a circuit at position i in a given list of circuits. """
    return circ_list[i].decompose(reps=10).depth()


tf = 2
dt = 0.05
time_axis = np.arange(0, tf+dt, dt)
n_qubits_axis = list(range(3,12))

tot_trot_n_err, tot_trot_dt_err, tot_pvqd_err, tot_pool_err = [], [], [], []
fin_trot_n_cnots, fin_trot_dt_cnots, fin_pvqd_cnots, fin_pool_cnots = [], [], [], []

for n_qubits in [str(i) for i in n_qubits_axis]:

    print(f'Processing simulations with {n_qubits} qubits.')

    # Get the time-evolved wavefunctions
    qutip_wvfs = np.squeeze(exact_datasets[n_qubits]['psi_t'])
    trot_n_wvfs = circs_to_vectors(trotter_fixed_depth_datasets[n_qubits]['evolved_states'])
    trot_dt_wvfs = circs_to_vectors(trotter_fixed_dt_datasets[n_qubits]['evolved_states']) 
    pvqd_wvfs = circs_to_vectors(pvqd_datasets[n_qubits]['evolved state']) 
    pool_wvfs = circs_to_vectors(pool_datasets[n_qubits]['evolved state'])

    # Total accumulated errors
    tot_trot_n_err.append(np.cumsum(1-exact_fidelity(qutip_wvfs, trot_n_wvfs))[-1]/len(time_axis))
    tot_trot_dt_err.append(np.cumsum(1-exact_fidelity(qutip_wvfs, trot_dt_wvfs))[-1]/len(time_axis))
    tot_pvqd_err.append(np.cumsum(1-exact_fidelity(qutip_wvfs, pvqd_wvfs))[-1]/len(time_axis))
    tot_pool_err.append(np.cumsum(1-exact_fidelity(qutip_wvfs, pool_wvfs))[-1]/len(time_axis))

    # Get the final number of CNOTs in the circuits
    fin_trot_n_cnots.append(measure_depth(trotter_fixed_depth_datasets[n_qubits]['evolved_states'], -1))
    fin_trot_dt_cnots.append(measure_depth(trotter_fixed_dt_datasets[n_qubits]['evolved_states'], -1))
    fin_pvqd_cnots.append(measure_depth(pvqd_datasets[n_qubits]['evolved state'], -1))
    fin_pool_cnots.append(measure_depth(pool_datasets[n_qubits]['evolved state'], -1))






##### Save data
save_data = {}

save_data["n_qubits_axis"]    = n_qubits_axis
save_data["tot_trot_n_err"]   = tot_trot_n_err
save_data["tot_trot_dt_err"]  = tot_trot_dt_err
save_data["tot_pvqd_err"]     = tot_pvqd_err 
save_data["tot_pool_err"]     = tot_pool_err

save_data["fin_trot_n_cnots"]  = fin_trot_n_cnots
save_data["fin_trot_dt_cnots"] = fin_trot_dt_cnots
save_data["fin_pvqd_cnots"]    = fin_pvqd_cnots
save_data["fin_pool_cnots"]    = fin_pool_cnots


file_save = "./driven_data_xyz.dat"
with open(file_save,'wb') as f:
    pickle.dump(save_data,f)
'''

##### Load data
file_save = "driven_xyz/driven_data_xyz.dat"
with open(file_save, 'rb') as f:
    save_data = pickle.load(f)


n_qubits_axis = save_data["n_qubits_axis"]    
tot_trot_n_err = save_data["tot_trot_n_err"]    
tot_trot_dt_err = save_data["tot_trot_dt_err"]   
tot_pvqd_err = save_data["tot_pvqd_err"]     
tot_pool_err = save_data["tot_pool_err"]      
fin_trot_n_cnots = save_data["fin_trot_n_cnots"]  
fin_trot_dt_cnots = save_data["fin_trot_dt_cnots"] 
fin_pvqd_cnots = save_data["fin_pvqd_cnots"]    
fin_pool_cnots = save_data["fin_pool_cnots"]    


# Instantiate a 2-panel plot
#fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw = {'hspace':0.05})
fig, ax = plt.subplots(2, 1, sharex=True,dpi=100,figsize=(3.4,5.1))
#fig.set_figheight(4.5)
#fig.set_figwidth(4)

# Color scheme
blue = ["#9ecae1","#3182bd"]  
red = ["#fc9272","#de2d26"]
green = ["#a1d99b","#31a354"]
gold = ["#FFD700","#DAA520"]


# Plot the total accumulated error (infidelity)
ax[0].plot(n_qubits_axis, tot_trot_dt_err, c=green[1], marker='^', ms=5, mfc=green[0], mec=green[1], mew=1, label='Trotter, $dt=0.05$')
ax[0].plot(n_qubits_axis, tot_trot_n_err, c=red[1], marker='p', ms=5, mfc=red[0], mec=red[1], mew=1, label='Trotter, $n_\mathrm{TS}=10$')
ax[0].plot(n_qubits_axis, tot_pvqd_err, c=gold[1], marker='s', ms=5, mfc=gold[0], mec=gold[1], mew=1, label='pVQD, $n_\mathrm{TS}=3$')
ax[0].plot(n_qubits_axis, tot_pool_err, c=blue[1], marker='o', ms=5, mfc=blue[0], mec=blue[1], mew=1, label='Adaptive pVQD')
ax[0].set_ylabel('$\Delta_\mathcal{I}^\mathrm{ex}(t_f)/t_f$')
ax[0].get_xaxis().set_visible(False)
ax[0].set_yscale('log')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), ncol=2,borderpad=0.5,handlelength=1.4,columnspacing=1.5,labelspacing=0.33)


# Plot the # of CNOTs in the circuits at the final time tf=2
ax[1].plot(n_qubits_axis, fin_trot_n_cnots, c=green[1], marker='^', ms=5, mfc=green[0], mec=green[1], mew=1,)
ax[1].plot(n_qubits_axis, fin_trot_dt_cnots, c=red[1], marker='p', ms=5, mfc=red[0], mec=red[1], mew=1)
ax[1].plot(n_qubits_axis, fin_pvqd_cnots, c=gold[1], marker='s', ms=5, mfc=gold[0], mec=gold[1], mew=1)
ax[1].plot(n_qubits_axis, fin_pool_cnots, c=blue[1], marker='o', ms=5, mfc=blue[0], mec=blue[1], mew=1)
# ax[1].set_ylabel('Final number of CNOTs')
ax[1].set_ylabel('Final depth')
ax[1].set_xlabel('Number of qubits')
ax[1].set_yscale('log')
ax[1].set_xticks(n_qubits_axis)

#plt.savefig("total_error_depth_vs_n_qubits_driven_xyz_params=1_0.8_0.6_0_bc=open.pdf", dpi=600, bbox_inches='tight')
plt.subplots_adjust(left=0.164,bottom=0.074,right=0.98,top=0.89,wspace=0.067,hspace=0.05)

plt.show()