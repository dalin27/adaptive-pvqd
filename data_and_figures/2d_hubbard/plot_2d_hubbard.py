import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
import pickle


os.chdir('data_and_figures/2d_hubbard')


## Color scheme
light_blue, blue = ["#9ecae1","#3182bd"]  
light_red, red = ["#fc9272","#de2d26"]
light_green, green = ["#a1d99b","#31a354"]
light_violet, violet = ["#DDA0DD","#BA55D3"] 


## Plot settings
ms = 20  # marker size
ew = 0.6  # marker edge width
rcParams.update({'font.size': 13})


## Load the datasets
exact_data = np.load('exact_2x2_params=1_0.8_bc=open.npz')
trotter_data = np.load('trotter_n=5_2x2_params=1_0.8_bc=open.npz')
maxiter = 400
ti = 0
tf = 6
block_filename = f'block_2x2_params=1_0.8_tol=0.99999_ST=2-2_mxit={maxiter}_ti={ti:.2f}_tf={tf:.2f}.pkl'
block_data = pickle.load(open(block_filename, 'rb'))
l_pool_filename = f'local_pool_2x2_params=1_0.8_tol=0.99995_ST=2-2_mxit={maxiter}_ti={ti:.2f}_tf={tf:.2f}.pkl'
l_pool_data = pickle.load(open(l_pool_filename, 'rb'))
nl_pool_filename = f'non-local_pool_2x2_params=1_0.8_tol=0.99997_ST=2-2_mxit={maxiter}_ti={ti:.2f}_tf={tf:.2f}.pkl'
nl_pool_data = pickle.load(open(nl_pool_filename, 'rb'))


## Get the time axes
exact_t = exact_data['times']
trotter_t = trotter_data['times']
block_t = block_data['times']
l_pool_t = l_pool_data['times']
nl_pool_t = nl_pool_data['times']


## Get the number of CNOTs of the (fully) decomposed parametrized circuit at each time step
trotter_cnots = np.full(len(trotter_t), trotter_data['cnots'])
block_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in block_data['evolved state'][1:]]
l_pool_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in l_pool_data['evolved state'][1:]]
nl_pool_cnots = [0]+[circ.decompose(reps=3).count_ops()["cx"] for circ in nl_pool_data['evolved state'][1:]]


## Get the expectation value of the observables of interest at each time step
block_n0n4, block_n0n2 = np.array(block_data["observables values"]).T
l_pool_n0n4, l_pool_n0n2 = np.array(l_pool_data["observables values"]).T
nl_pool_n0n4, nl_pool_n0n2 = np.array(nl_pool_data["observables values"]).T


## Plot the data
fig, ax = plt.subplots(3, 1, figsize=(6,8))

# Top panel to plot <n0n4>
ax[0].plot(exact_t, exact_data['n0n4'], ls='--', linewidth=1, color='k',)
ax[0].scatter(trotter_t, trotter_data['n0n4'], s=ms, marker='p', c=light_red, linewidths=ew, edgecolors=red)
ax[0].scatter(block_t, block_n0n4, s=ms, marker='o', c=light_violet, linewidths=ew, edgecolors=violet)
ax[0].scatter(l_pool_t, l_pool_n0n4, s=ms, marker='o', c=light_blue, linewidths=ew, edgecolors=blue)
ax[0].scatter(nl_pool_t, nl_pool_n0n4, s=ms, marker='^', c=light_green, linewidths=ew, edgecolors=green)
ax[0].set(ylabel='$\\langle n_{0 \\uparrow } \, n_{0  \\downarrow } \\rangle$')
ax[0].get_xaxis().set_visible(False)

# Middle panel to plot <n0n2>
ax[1].plot(exact_t, exact_data['n0n2'], ls='--', linewidth=1, color='k')
ax[1].scatter(trotter_t, trotter_data['n0n2'], s=ms, marker='p', c=light_red, linewidths=ew, edgecolors=red)
ax[1].scatter(block_t, block_n0n2, s=ms, marker='o', c=light_violet, linewidths=ew, edgecolors=violet)
ax[1].scatter(l_pool_t, l_pool_n0n2, s=ms, marker='o', c=light_blue, linewidths=ew, edgecolors=blue)
ax[1].scatter(nl_pool_t, nl_pool_n0n2, s=ms, marker='^', c=light_green, linewidths=ew, edgecolors=green)
ax[1].set(ylabel='$\\langle n_{0 \\uparrow } \, n_{2  \\uparrow } \\rangle$')
ax[1].get_xaxis().set_visible(False)

# Bottom panel to plot the number of CNOTs 
trotter_instance = ax[2].scatter(trotter_t, trotter_cnots, s=ms, marker='p', c=light_red, linewidths=ew, edgecolors=red)
block_instance = ax[2].scatter(block_t, block_cnots, s=ms, marker='o', c=light_violet, linewidths=ew, edgecolors=violet)
l_pool_instance = ax[2].scatter(l_pool_t, l_pool_cnots, s=ms, marker='o', c=light_blue, linewidths=ew, edgecolors=blue)
nl_pool_instance = ax[2].scatter(nl_pool_t, nl_pool_cnots, s=ms, marker='^', c=light_green, linewidths=ew, edgecolors=green)


ax[2].set_xlabel('$Jt$')
ax[2].set_ylabel('CNOTs')


ex = Line2D([], [], lw=1, label='Exact', color='k', linestyle='--')
ax[0].legend(handles=[ex, trotter_instance, block_instance, l_pool_instance, nl_pool_instance], 
            labels=['Exact', 'Trotter, $n_\mathrm{TS}=5$', 'pVQD, block', 'Adaptive pVQD, L', 'Adaptive pVQD, NL'],
            loc='upper center', bbox_to_anchor=(0.498, 1.5), ncol=2, columnspacing=0.94,labelspacing=0.1,borderpad=0.5,handlelength=1)

outfilename = 'adaptive_pvqd_2x2_params=1_0.8.pdf'
# plt.savefig(outfilename, dpi=400, bbox_inches='tight')
plt.show()