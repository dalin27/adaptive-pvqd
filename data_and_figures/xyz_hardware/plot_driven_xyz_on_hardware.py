# Import the relevant libraries
import os
import pickle
import numpy as np


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


#os.chdir('data_and_figures/driven_xyz_on_hardware')

## Exact dataset
exact_data = np.load('exact_xyz_floquet_n_qubits=4_params=1_0.8_0.6_0_bc=open.npz')

## Trotter datasets
# Noisy:
noisy_trotter_datasets = []
noisy_trotter_dir = 'manila_noisy_znequad123_dd_trotter/'
for filename in os.listdir(noisy_trotter_dir):
    with open(noisy_trotter_dir+filename, 'rb') as f:
        noisy_trotter_datasets.append(pickle.load(f))
# Hardware:
hardware_trotter_datasets = []
hardware_trotter_dir = 'manila_hardware_znequad123_dd_trotter/'
for filename in os.listdir(hardware_trotter_dir):
    with open(hardware_trotter_dir+filename, 'rb') as f:
        hardware_trotter_datasets.append(pickle.load(f))

## Pool Adaptive pVQD datasets
# Noisy:
noisy_pool_adap_datasets = []
noisy_pool_adap_dir = 'manila_noisy_znequad123_dd_pool/'
for filename in os.listdir(noisy_pool_adap_dir):
    with open(noisy_pool_adap_dir+filename, 'rb') as f:
        noisy_pool_adap_datasets.append(pickle.load(f))
# Hardware:
hardware_pool_adap_datasets = []
hardware_pool_adap_dir = 'manila_hardware_znequad123_dd_pool/'
for filename in os.listdir(hardware_pool_adap_dir):
    with open(hardware_pool_adap_dir+filename, 'rb') as f:
        hardware_pool_adap_datasets.append(pickle.load(f))

# Get the time axes
times = np.arange(0, 2.05, 0.05)

# Instantiate the 3-panel plot
fig, ax = plt.subplots(2, 1, sharex=True,dpi=100,figsize=(3.4,5.1))#, gridspec_kw = {'hspace':0.05})
#fig.set_figheight(4)
#fig.set_figwidth(4)




# Color scheme
green = ["#a1d99b","#31a354"] 
blue = ["#9ecae1","#3182bd"]

# Noisy:
noisy_trot_z0s = [np.array(x[0]) for x in noisy_trotter_datasets]
noisy_trot_z0_max = np.max(noisy_trot_z0s, axis=0)
noisy_trot_z0_min = np.min(noisy_trot_z0s, axis=0)

noisy_trot_z0z1s = [np.array(x[1]) for x in noisy_trotter_datasets]
noisy_trot_z0z1_max = np.max(noisy_trot_z0z1s, axis=0)
noisy_trot_z0z1_min = np.min(noisy_trot_z0z1s, axis=0)

noisy_pool_z0s = [np.array(x[0]) for x in noisy_pool_adap_datasets]
noisy_pool_z0_max = np.max(noisy_pool_z0s, axis=0)
noisy_pool_z0_min = np.min(noisy_pool_z0s, axis=0)

noisy_pool_z0z1s = [np.array(x[1]) for x in noisy_pool_adap_datasets]
noisy_pool_z0z1_max = np.max(noisy_pool_z0z1s, axis=0)
noisy_pool_z0z1_min = np.min(noisy_pool_z0z1s, axis=0)


# Hardware:
hard_trot_z0s = [np.array(x[0]) for x in hardware_trotter_datasets]
hard_trot_z0_mean = np.mean(hard_trot_z0s, axis=0)
hard_trot_z0_std = np.std(hard_trot_z0s, axis=0)

hard_trot_z0z1s = [np.array(x[1]) for x in hardware_trotter_datasets]
hard_trot_z0z1_mean = np.mean(hard_trot_z0z1s, axis=0)
hard_trot_z0z1_std = np.std(hard_trot_z0z1s, axis=0)

hard_pool_z0s = [np.array(x[0]) for x in hardware_pool_adap_datasets]
hard_pool_z0_mean = np.mean(hard_pool_z0s, axis=0)
hard_pool_z0_std = np.std(hard_pool_z0s, axis=0)

hard_pool_z0z1s = [np.array(x[1]) for x in hardware_pool_adap_datasets]
hard_pool_z0z1_mean = np.mean(hard_pool_z0z1s, axis=0)
hard_pool_z0z1_std = np.std(hard_pool_z0z1s, axis=0)


### Top panel to plot <z0>
## Trotter 
trot_noisy=ax[0].fill_between(times[0::4], noisy_trot_z0_min, noisy_trot_z0_max, facecolor=green[0],
                              edgecolor=green[1], linewidth=0.3, hatch='||', alpha=0.25)
trot_hard=ax[0].errorbar(times[0::4], hard_trot_z0_mean, hard_trot_z0_std, marker='^', markersize=4,
                        c=green[0], mec=green[1], ecolor=green[1], mew=0.6, fmt=' ', capsize=1.5, elinewidth=0.5, capthick=0.5)

## Pool Adaptive pVQD
pool_noisy=ax[0].fill_between(times[0::4], noisy_pool_z0_min, noisy_pool_z0_max, facecolor=blue[0],
                              edgecolor=blue[1], linewidth=0.3, hatch='..', alpha=0.25)
pool_hard=ax[0].errorbar(times[0::4], hard_pool_z0_mean, hard_pool_z0_std, marker='o', markersize=4,
                        c=blue[0], mec=blue[1], ecolor=blue[1], mew=0.6, fmt=' ', capsize=1.5, elinewidth=0.5, capthick=0.5)

ax[0].plot(exact_data['times'], exact_data['z0'], ls='--', linewidth=1, color='k')
ax[0].set(ylabel='$\\langle Z_0 \\rangle$')
ax[0].set_xlim(xmin=0, xmax=2)
ax[0].get_xaxis().set_visible(False)

### Bottom panel to plot <z0z1>
## Trotter 
ax[1].fill_between(times[0::4], noisy_trot_z0z1_min, noisy_trot_z0z1_max, facecolor=green[0],
                              edgecolor=green[1], linewidth=0.3, hatch='||', alpha=0.25)
ax[1].errorbar(times[0::4], hard_trot_z0z1_mean, hard_trot_z0z1_std, marker='^', markersize=4,
                        c=green[0], mec=green[1], ecolor=green[1], mew=0.6, fmt=' ', capsize=1.5, elinewidth=0.5, capthick=0.5)

## Pool Adaptive pVQD
ax[1].fill_between(times[0::4], noisy_pool_z0z1_min, noisy_pool_z0z1_max, facecolor=blue[0],
                              edgecolor=blue[1], linewidth=0.3, hatch='..', alpha=0.25)
ax[1].errorbar(times[0::4], hard_pool_z0z1_mean, hard_pool_z0z1_std, marker='o', markersize=4,
                        c=blue[0], mec=blue[1], ecolor=blue[1], mew=0.6, fmt=' ', capsize=1.5, elinewidth=0.5, capthick=0.5)

ax[1].plot(exact_data['times'], exact_data['z0z1'], ls='--', linewidth=1, color='k')
ax[1].set(ylabel='$\\langle Z_0 Z_1 \\rangle$')
ax[1].set_xlim(xmin=0, xmax=2)
ax[1].set_xlabel('$J_x t$')

ex = Line2D([], [], lw=1, label='Exact', color='k', linestyle='--')
empty = Line2D([], [], label=' ', linestyle=' ')
lg = ax[0].legend(handles=[ex, empty, trot_noisy, trot_hard, empty, empty, pool_noisy, pool_hard], 
            labels=['Exact','','Noise model','Hardware',' ',
                    '','Noise model','Hardware'],
            loc='upper center', bbox_to_anchor=(0.5, 1.59), ncol=2, columnspacing=4.1,
            handleheight=1.5, labelspacing=0.35,borderpad=0.75)
#[lg.get_texts()[i].set_x(-17) for i in [1,5]]
#[lg.get_texts()[i].set_x(-35) for i in [1,5]]

ax[0].text(0.07, 1.83, 'Trotter, $dt=0.2$:', fontsize = 9,zorder=10000)
ax[0].text(1.18, 1.83, 'Adaptive pVQD:', fontsize = 9,zorder=10000)

#axd["A"].legend(loc="upper center", bbox_to_anchor=(0.777, 1.35),ncol=3,fancybox=False,numpoints=1,handlelength=1,columnspacing=1.1,handletextpad=0.4,borderpad=0.75)

#plt.rcParams.update({'font.size': 2})
ax[1].set_xticks([0,0.5,1,1.5,2])
ax[0].set_yticks([-0.5,0,0.5,1])
ax[1].set_yticks([0,-0.5,-1])
# plt.savefig("hardware_xyz_floquet_n_qubits=4_params=1_0.8_0.6_0_bc=open_tol=0.9999_shots=1e5.pdf", dpi=600, bbox_inches='tight')

plt.subplots_adjust(left=0.16,bottom=0.074,right=0.97,top=0.8,wspace=0.067,hspace=0.05)
plt.show()

