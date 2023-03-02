## Adaptive Projected Variational Quantum Dynamics (Adaptive pVQD) 

Implementatiton based on: 
- https://github.com/StefanoBarison/p-VQD
- https://github.com/Qiskit/qiskit-terra/tree/main/qiskit/algorithms/time_evolvers/pvqd

which are in turn mainly based on the following papers:

Standard pVQD:
Stephano Barison, Filippo Vicentini and Giuseppe Carleo, An efficient quantum algorithm for the 
time evolution of parameterized circuits, 2021 (https://arxiv.org/abs/2101.04579).

Adaptive pVQD:
[LINK TO THE PAPER]

### Content of the repository

- **main.py** : run the Adaptive pvQD algorithm for a given Hamiltonian and initial conditions;
- **hamiltonian.py** : define different time-dependent and time-independent Hamiltonians;
- **parametrized_circuit.py** : append initial conditions (and possibly a starting ansatz) to a circuit; 
- **utils.py**: define utility functions;
- **adaptive_pvqd.py** : evolve in time the system of interest (backbone of the code);
- **exact_evol_with_qutip.py** : "exact" quantum dynamics using QuTiP (solving the Lindblad master equation);
- **trotter_evolution.py** : implement the Trotter evolution of a given Hamiltonian;
- **data_and_figures** : folder that contains pre-produced results of simulations, associated plots and the Python scripts used to produce the plots


