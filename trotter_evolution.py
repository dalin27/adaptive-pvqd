# Import the relevant libraries
from hamiltonian import *
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import StateFn, CircuitSampler, ListOp, PauliExpectation
import numpy as np


def trotterized_hamiltonian(hamiltonian, trotter_steps, final_time):
    '''
    Circuit implementing Trotterization of the time evolutiom operator for a given Hamiltonian
        
    Args:
        hamiltonian (SumPauliOp): Hamiltonian to Trotterize
        trotter_steps (int): number of trotter steps to implement
        final_time (float): total simulation time
        
    Returns:
        QuantumCircuit implementing the Trotterization of the time evolutiom operator
    '''

    dt = final_time/trotter_steps
    qc = QuantumCircuit(n_qubits) 

    # Loop over the number of Trotter steps
    for _ in range(trotter_steps):
        # Loop over the terms in the Hamiltonian
        for hi in hamiltonian:
            qc_new = QuantumCircuit(n_qubits)
            qc_new = qc_new.compose(PauliEvolutionGate(operator=hi, time=dt), range(n_qubits))
            qc = qc.compose(qc_new)
        qc.barrier()

    return qc


# Define Qiskit evaluating tools
backend = Aer.get_backend("statevector_simulator")
sampler = CircuitSampler(backend)
expectation = PauliExpectation() 

# Fix some global variables
n_qubits = 8  # number of qubits
tf = 2  # final time of the simulation
times = np.arange(0, tf, 0.05)  # range of times for the simulation
trotter_steps = 20  # number of trotter steps

## Model parameters

# # 2D Hubbard model
# model_params = [1,0.8]  # hopping and Coulomb parameters
# lx, ly = 2, 2  # size of the square grid

# XYZ Floquet model
model_params = [1,1,1,0]

## Define the initial state
psi0 = QuantumCircuit(n_qubits)

# # 2D Hubbard model half-filled antiferromagnetic state
# fermion_idx = [0,3,4,7]  # !! HARDCODED for a 2x2 square lattice !!
# psi0.x(fermion_idx)
# psi0.barrier()

# 1D XYZ Floquet model antiferromagnetic state
odd_idx = list(range(n_qubits))[1::2]
psi0.x(odd_idx)
psi0.barrier()

## List of observables to measure at each time t of the simulation
# observables = ListOp([num_op(n_qubits, 0) @ num_op(n_qubits, 1),
#                num_op(n_qubits, 0) @ num_op(n_qubits, 4)])
observables = ListOp([one_site_op(n_qubits, 0, X),
               one_site_op(n_qubits, 0, Z),
               nearest_neigh_op(n_qubits, 0, [X, X]),
               nearest_neigh_op(n_qubits, 0, [Z, Z])])

# Simulate the time evolution of psi0 by looping over multiple times:
obs_values = []
for t in times:

    # Get the Hamiltonian at time t 
    # hamiltonian = hubbard_2D_hamiltonian(size_x=lx, size_y=ly, model_params=model_params)
    hamiltonian = xyz_floquet_hamiltonian(n_qubits, model_params, time=t)

    # Prepare the Hamiltonian at the given time t
    qc_trot = trotterized_hamiltonian(
        hamiltonian=hamiltonian,
        trotter_steps=trotter_steps,
        final_time=t
    )

    # Apply the trotterized Hamiltonian on the initial condition and perform measurements on the time evolved state
    evolved_circ_state = psi0.compose(qc_trot)  
    expectation_values = StateFn(observables, is_measurement=True) @ StateFn(evolved_circ_state)
    expectation_values = expectation.convert(expectation_values)
    sampled_op = sampler.convert(expectation_values)
    obs_values.append(np.real(sampled_op.eval()))

# Get the depth and the number of CNOTs in the Trotter circuit (to be used when plotting the results)
decomposed_circ = qc_trot.decompose(reps=10)
depth = decomposed_circ.depth()
cnots = decomposed_circ.count_ops()["cx"]
print(f'Final circuit: {decomposed_circ}')
print(f'Depth of the circuit: {depth}')
print(f'Number of CNOTs: {cnots}')

## Save the data in a file (!! CAREFUL, file name HARDCODED !!)

# # Hubbard model
# n0n1, n0n4 = np.array(obs_values).T
# np.savez(
#     'data_and_figures/hubbard/trotter_evolution_n=5_hubbard_2x2_params=1_0.8_bc=open.npz',
#     times=times,
#     depth=depth,
#     cnots=cnots,
#     n0n1=n0n1,
#     n0n4=n0n4 
# )

# XYZ Floquet model
x0, z0, x0x1, z0z1 = np.array(obs_values).T
np.savez(
    'data_and_figures/xyz_floquet/trotter_evolution_n=20_xyz_floquet_n_qubits=8_params=1_1_1_0_bc=open.npz',
    times=times,
    depth=depth,
    cnots=cnots,
    x0=x0,
    z0=z0,
    x0x1=x0x1,
    z0z1=z0z1
)
