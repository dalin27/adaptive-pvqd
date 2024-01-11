## Adaptive Projected Variational Quantum Dynamics (Adaptive pVQD) 

Repository associated with the Adaptive pVQD paper:
David Linteau, Stefano Barison, Netanel Lindner, Giuseppe Carleo, Adaptive projected variational quantum dynamics,
2023, (https://arxiv.org/abs/2307.03229).

Implementatiton based on: 
- https://github.com/StefanoBarison/p-VQD
- https://github.com/qiskit-community/qiskit-algorithms/tree/main/qiskit_algorithms/time_evolvers/pvqd

### Content of the repository

- **main.py** : run the Adaptive pvQD algorithm for a given Hamiltonian and initial conditions;
- **hamiltonian.py** : define different time-dependent and time-independent Hamiltonians;
- **parametrized_circuit.py** : append initial conditions (and possibly a starting ansatz) to a circuit; 
- **utils.py**: define utility functions;
- **adaptive_pvqd.py** : evolve in time the system of interest (backbone of the code);
- **exact_evol_with_qutip.py** : "exact" quantum dynamics using QuTiP (solving the Lindblad master equation);
- **trotter_evolution.py** : implement the Trotter evolution of a given Hamiltonian;
- **data_and_figures** : folder that contains pre-produced results of simulations, associated plots and the Python scripts used to produce the plots


