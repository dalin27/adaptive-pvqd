# Import the relevant libraries
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import X, Z, PauliExpectation
from hamiltonian import *
from parametrized_circuit import * 
from adaptive_pvqd import AdaptivePVQD
from functools import partial


# Define some relevant variables
n_qubits = 8
tf = 2  # final time
dt = 0.05  # time step

# Backend choice (statevector simulation)
backend = Aer.get_backend("statevector_simulator") 
instance = QuantumInstance(backend=backend, shots=1)

## Define a (possibly time-dependent) Hamiltonian
# hamiltonian = partial(hubbard_ladder_hamiltonian, size_x=2, model_params=[1,0.8])
hamiltonian = partial(xyz_floquet_hamiltonian, n_qubits=n_qubits, model_params=[1,0.8,0.6,0])

## Define the initial ansatz and parameters (if applicable)
# ansatz = AnsatzLadderHubbard(n_qubits, depth=0)
ansatz = AnsatzXYZFloquet(n_qubits, depth=3)
initial_parameters = np.zeros(ansatz.circuit.num_parameters)

# List of observables to measure at each time t of the simulation
# observables = [num_op(n_qubits, 0) @ num_op(n_qubits, 4),
#                num_op(n_qubits, 0) @ num_op(n_qubits, 2)]
observables=[one_site_op(n_qubits, 0, X),
            one_site_op(n_qubits, 0, Z),
            nearest_neigh_op(n_qubits, 0, [X, X]),
            nearest_neigh_op(n_qubits, 0, [Z, Z])]


# Initialize the algorithm
adaptive_pvqd = AdaptivePVQD(
    ansatz,
    maxiter=200,
    fidelity_tolerance=0.9999,
    gradient_tolerance=5e-5,
    expectation= PauliExpectation(),
    quantum_instance=instance,
)

# Evolve in time
result = adaptive_pvqd.evolve(
        hamiltonian,
        num_time_steps=int(np.ceil(tf/dt)),
        final_time=tf,
        initial_parameters=initial_parameters,
        shift_init_guess=np.zeros(len(initial_parameters)),
        observables=observables
)

# # Save the data in a file
# import pickle
# # f = open("data_and_figures/2d_hubbard/pool_adaptive_pvqd_hubbard_2x2_params=1_0.8_n=2_bc=open_tol=0.9999.pkl", "wb")
# f = open("data_and_figures/driven_xyz/pvqd/pvqd_n=3_driven_xyz_n_qubits=8_params=1_0.8_0.6_0_bc=open.pkl", "wb")
# pickle.dump(result, f)
# f.close()

