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
# hamiltonian = partial(hubbard_2D_hamiltonian, size_x=2, size_y=2, model_params=[1,0.8])
hamiltonian = partial(xyz_floquet_hamiltonian, n_qubits=n_qubits, model_params=[1,1,1,0])

## Define the initial ansatz and parameters (if applicable)
# ansatz = AnsatzHubbard2D(n_qubits)
ansatz = AnsatzXYZFloquet(n_qubits)
initial_parameters = np.zeros(ansatz.circuit.num_parameters)

## List of observables to measure at each time t of the simulation
# observables = [num_op(n_qubits, 0) @ num_op(n_qubits, 1),
#                num_op(n_qubits, 0) @ num_op(n_qubits, 4)]
observables=[one_site_op(n_qubits, 0, X),
             one_site_op(n_qubits, 0, Z),
             nearest_neigh_op(n_qubits, 0, [X, X]),
             nearest_neigh_op(n_qubits, 0, [Z, Z])]

# Initialize the algorithm
adaptive_pvqd = AdaptivePVQD(
    ansatz,
    maxiter=400,
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

# Save the data in a file
import pickle
# f = open("data_and_figures/hubbard/adaptive_pvqd_hubbard_2x2_params=1_0.8_bc=open_tol=0.9999.pkl", "wb")
f = open("data_and_figures/xyz_floquet/adaptive_pvqd_xyz_floquet_params=1_1_1_0_bc=open_tol=0.9999.pkl", "wb")
pickle.dump(result, f)
f.close()

