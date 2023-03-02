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

#os.chdir('data_and_figures/2d_hubbard')

# Load the datasets
exact_data = np.load('exact_hubbard_2x2_params=1_0.8_bc=open.npz')
trotter_data = np.load('trotter_evolution_n=5_hubbard_2x2_params=1_0.8_bc=open.npz')
# with open('trotter_adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl', 'rb') as f:
#     trotter_adaptive_pvqd_data = pickle.load(f)
with open('pool_adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl', 'rb') as f:
    pool_adaptive_pvqd_data = pickle.load(f)
with open('long_range_pool_adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl', 'rb') as f:
    lr_pool_adaptive_pvqd_data = pickle.load(f)
# with open('mixed_trot_pool_adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl', 'rb') as f:
#     mixed_pvqd_data = pickle.load(f)

# Get the time axes
trot_t = trotter_data["times"]
# trot_adap_t = trotter_adaptive_pvqd_data["times"]
pool_adap_t = pool_adaptive_pvqd_data["times"]
lr_pool_adap_t = lr_pool_adaptive_pvqd_data["times"]
# mixed_t = mixed_pvqd_data["times"]

# Get the expectation value of the observables of interest at each time step
# trot_adap_n0n4, trot_adap_n0n2 = np.array(trotter_adaptive_pvqd_data["observables values"]).T
pool_adap_n0n4, pool_adap_n0n2 = np.array(pool_adaptive_pvqd_data["observables values"]).T
lr_pool_adap_n0n4, lr_pool_adap_n0n2 = np.array(lr_pool_adaptive_pvqd_data["observables values"]).T
# mixed_n0n4, mixed_n0n2 = np.array(mixed_pvqd_data["observables values"]).T

# Get the number of CNOTs and the depth of the (fully) decomposed parametrized circuit at each time step
trot_cnots = np.full(len(trot_t), trotter_data['cnots'])
# trot_adap_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in trotter_adaptive_pvqd_data['evolved state'][1:]]
pool_adap_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in pool_adaptive_pvqd_data['evolved state'][1:]]
lr_pool_adap_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in lr_pool_adaptive_pvqd_data['evolved state'][1:]]
# mixed_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in mixed_pvqd_data['evolved state'][1:]]

# Instantiate the 3-panel plot
#fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw = {'hspace':0.05})
fig, ax = plt.subplots(3, 1, sharex=True,dpi=100,figsize=(3.4,5.1))
#fig.set_figheight(4.8)
#fig.set_figwidth(4)

# Color scheme
blue = ["#9ecae1","#3182bd"]  
red = ["#fc9272","#de2d26"]
green = ["#a1d99b","#31a354"]
violet = ["#DDA0DD","#BA55D3"] 
gold = ["#FFD700","#DAA520"]


# Top panel to plot <n0n4>
ax[0].scatter(trot_t, trotter_data['n0n4'], s=10,      marker='p', c=red[0], linewidths=0.6, edgecolors=red[1])
ax[0].scatter(lr_pool_adap_t, lr_pool_adap_n0n4, s=10, marker='s', c=gold[0], linewidths=0.6, edgecolors=gold[1])
ax[0].scatter(pool_adap_t, pool_adap_n0n4, s=8,        marker='o', c=blue[0], linewidths=0.6, edgecolors=blue[1])

ax[0].plot(exact_data['times'], exact_data['n0n4'], ls='--', linewidth=1, color='k')
ax[0].set(ylabel='$\\langle n_{0 \\uparrow } \, n_{0  \\downarrow } \\rangle$')
ax[0].set_xlim(xmin=-0.1, xmax=4.1)
ax[0].get_xaxis().set_visible(False)

# Middle panel to plot <n0n2>
ax[1].scatter(trot_t, trotter_data['n0n2'], s=8,       marker='p', c=red[0], linewidths=0.6, edgecolors=red[1])
ax[1].scatter(lr_pool_adap_t, lr_pool_adap_n0n2, s=10, marker='s', c=gold[0], linewidths=0.6, edgecolors=gold[1])
ax[1].scatter(pool_adap_t, pool_adap_n0n2, s=8,        marker='o', c=blue[0], linewidths=0.6, edgecolors=blue[1])

ax[1].plot(exact_data['times'], exact_data['n0n2'], ls='--', linewidth=1, color='k')
ax[1].set(ylabel='$\\langle n_{0 \\uparrow } \, n_{2  \\uparrow } \\rangle$')
#ax[1].set_xlim(xmin=0, xmax=4)
ax[1].get_xaxis().set_visible(False)

# Bottom panel to plot the number of CNOTs and the depth
trot=ax[2].scatter(trot_t, trot_cnots, s=10,                       marker='p', c=red[0], linewidths=0.6, edgecolors=red[1])
lr_pool_ad=ax[2].scatter(lr_pool_adap_t, lr_pool_adap_cnots, s=10, marker='s', c=gold[0], linewidths=0.6, edgecolors=gold[1])
pool_ad=ax[2].scatter(pool_adap_t, pool_adap_cnots, s=8,           marker='o', c=blue[0], linewidths=0.6, edgecolors=blue[1])

ax[2].set_xlabel('$Jt$')
ax[2].set_ylabel('CNOTs')

ex = Line2D([], [], lw=1, label='Exact', color='k', linestyle='--')
# empty = Line2D([], [], label=' ', linestyle=' ')
ax[0].legend(handles=[ex, trot, pool_ad, lr_pool_ad], 
            labels=['Exact', 'Trotter, $n_\mathrm{TS}=5$', 'Adaptive pVQD, L', 'Adaptive pVQD, NL'],
            loc='upper center', bbox_to_anchor=(0.498, 1.38), ncol=2, columnspacing=0.94,labelspacing=0.1,borderpad=0.5,handlelength=1)

plt.rcParams.update({'font.size': 2})
ax[2].set_xticks([0,0.8,1.6,2.4,3.2,4])
ax[2].set_yticks([0,120,240])
ax[1].set_yticks([0,0.5,1])
ax[0].set_yticks([0,0.1,0.2])
#plt.savefig("adaptive_pvqd_schemes_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pdf", dpi=600, bbox_inches='tight')

plt.subplots_adjust(left=0.16,bottom=0.074,right=0.98,top=0.905,wspace=0.067,hspace=0.05)

plt.show()