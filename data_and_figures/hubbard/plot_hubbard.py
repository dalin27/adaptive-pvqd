# Import the relevant libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


os.chdir('data_and_figures/hubbard')

# Load the datasets
exact_data = np.load('exact_hubbard_2x2_params=1_0.8_bc=open.npz')
trotter_data = np.load('trotter_evolution_n=5_hubbard_2x2_params=1_0.8_bc=open.npz')
with open('adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl', 'rb') as f:
    adaptive_pvqd_data = pickle.load(f)

# Get the time axes
pvqd_t = adaptive_pvqd_data["times"]
trot_t = trotter_data["times"]

# Get the expectation value of the observables of interest at each time step
pvqd_n0n1, pvqd_n0n4 = np.array(adaptive_pvqd_data["observables values"]).T

# Get the number of CNOTs and the depth of the (fully) decomposed parametrized circuit at each time step
pvqd_cnots = [circ.decompose(reps=10).count_ops()["cx"] for circ in adaptive_pvqd_data['evolved state']]
pvqd_depths = [circ.decompose(reps=10).depth() for circ in adaptive_pvqd_data['evolved state']]
trot_depth = np.full(len(trot_t[1:]), trotter_data['depth'])
trot_cnots = np.full(len(trot_t[1:]), trotter_data['cnots'])

# Instantiate the 3-panel plot
fig, ax = plt.subplots(3, 1, sharex=True)
fig.set_figheight(5)
fig.set_figwidth(4)

# Color scheme
colors1 = ["#9ecae1","#3182bd"]  # blue
colors2 = ["#fc9272","#de2d26"]  # red
colors3 = ["#bdbdbd","#636363"]  # black

# Top panel to plot <n0n1>
ax[0].scatter(pvqd_t, pvqd_n0n1, s=8, marker='o', c=colors1[0], linewidths=0.8, edgecolors=colors1[1])
ax[0].scatter(trot_t, trotter_data["n0n1"], s=8, marker='s', c=colors2[0], linewidths=0.8, edgecolors=colors2[1])
ax[0].plot(exact_data['times'], exact_data['n0n1'], ls='--', linewidth=1, color='k')
ax[0].set(ylabel='$\\langle n_{0 \! \! \\uparrow \! \!} n_{0 \! \! \\downarrow \! \!} \\rangle$')
ax[0].set_xlim(xmin=0, xmax=2)

# Middle panel to plot <n0n4>
ax[1].scatter(pvqd_t, pvqd_n0n4, s=8, marker='o', c=colors1[0], linewidths=0.8, edgecolors=colors1[1])
ax[1].scatter(trot_t, trotter_data["n0n4"], s=8, marker='s', c=colors2[0], linewidths=0.8, edgecolors=colors2[1])
ax[1].plot(exact_data['times'], exact_data['n0n4'], ls='--', linewidth=1, color='k')
ax[1].set(ylabel='$\\langle n_{0 \! \! \\uparrow \! \!} n_{2 \! \! \\uparrow \! \!} \\rangle$')
ax[1].set_xlim(xmin=0, xmax=2)

# Bottom panel to plot the number of CNOTs and the depth
a20=ax[2].scatter(pvqd_t[1:], np.cumsum(pvqd_depths), s=8, marker='o', c='white', linewidths=0.8, edgecolors=colors1[1])
a21=ax[2].scatter(trot_t[1:], np.cumsum(trot_depth), s=8, marker='s', c='white', linewidths=0.8, edgecolors=colors2[1])
a22=ax[2].scatter(pvqd_t[1:], np.cumsum(pvqd_cnots), s=8, marker='o', c=colors1[0], linewidths=0.8, edgecolors=colors1[1])
a23=ax[2].scatter(trot_t[1:], np.cumsum(trot_cnots), s=8, marker='s', c=colors2[0], linewidths=0.8, edgecolors=colors2[1])
ax[2].set_xlabel('$Jt$')
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
ax[2].yaxis.set_major_formatter(mticker.FuncFormatter(g))

ax[2].legend(handles=[(a20,a21),(a22,a23)], labels=['Cumulative depth','Cumulative CNOTs'],
            handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=8)

a02 = Line2D([], [], lw=1, label='Exact', color='k', linestyle='--')
ax[0].legend(handles=[a02,(a20,a22),(a21,a23)], labels=['Exact','Adaptive pVQD','Trotter ($n=5$)'],
            loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=8,
            handler_map={tuple: HandlerTuple(ndivide=None)})

plt.tight_layout()
# plt.savefig("adaptive_pvqd_vs_trotter_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pdf", dpi=600)

plt.show()