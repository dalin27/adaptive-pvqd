# Import the relevant libraries
from hamiltonian import *
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import StateFn, CircuitSampler, ListOp, PauliExpectation
import numpy as np
from parametrized_circuit import *
from functools import partial
from qiskit.utils import QuantumInstance


def trotterized_hamiltonian(hamiltonian, trotter_steps, final_time):
    '''
    Circuit implementing Trotterization of the time evolutiom operator for a given Hamiltonian.
    This is a non-targeted (i.e. not resource efficient) way of implementing Trotterization.
        
    Args:
        hamiltonian (SumPauliOp): Hamiltonian to Trotterize
        trotter_steps (int): number of trotter steps to implement
        final_time (float): total simulation time
        
    Returns:
        QuantumCircuit implementing the Trotterization of the time evolutiom operator
    '''

    dt = final_time/trotter_steps
    qc = QuantumCircuit(n_qubits) 

    # # Get the time-independent terms of the (XYZ + drive term) model and make the circuit compact
    # ham = hamiltonian(time=0)
    # xx_yy_zz_terms = ham[0:3*(n_qubits-1)]
    # xx_yy_zz_terms_split = [xx_yy_zz_terms[i:i+3] for i in range(0, len(xx_yy_zz_terms), 3)]
    # xx_yy_zz_terms_compact = xx_yy_zz_terms_split[0::2] + xx_yy_zz_terms_split[1::2]

    # Loop over the number of Trotter steps
    for ts in range(trotter_steps):

        # # Time-independent terms (same terms and same coefficients added for each Trotter step)
        # for term in xx_yy_zz_terms_compact:
        #     qc = qc.compose(PauliEvolutionGate(operator=term, time=dt), range(n_qubits))

        # # Time-dependent terms (same terms but different coefficients added for each Trotter step)
        # ham = hamiltonian(time=ts*dt)
        # if len(ham) != 3*(n_qubits-1):  # i.e. if final_time > 0:
        #     for term in [ham[3*(n_qubits-1):]]:
        #         qc = qc.compose(PauliEvolutionGate(operator=term, time=dt), range(n_qubits))

        qc = qc.compose(PauliEvolutionGate(operator=hamiltonian, time=dt), range(n_qubits))

        qc.barrier()

    return qc


# Statevector simulations:
backend = Aer.get_backend("statevector_simulator")
sampler = CircuitSampler(backend)
expectation = PauliExpectation() 
shots = 0

# # Noisy simulations
# from qiskit.opflow import StateFn, ListOp, CircuitSampler, PauliExpectation
# from qiskit import IBMQ
# from qiskit.utils import QuantumInstance
# provider = IBMQ.load_account()
# backend = provider.get_backend('ibm_oslo')
# from qiskit_aer.backends import QasmSimulator
# backend = QasmSimulator.from_backend(backend)
# shots = 10000
# instance = QuantumInstance(backend=backend, shots=shots)
# sampler = CircuitSampler(instance)
# expectation = PauliExpectation()


# # (Real) hardware simulations
# shots = 10000
# from qiskit import IBMQ
# # from qiskit.providers.ibmq import least_busy
# if not IBMQ.active_account():
#     IBMQ.load_account()
# provider = IBMQ.get_provider()
# # backend = least_busy(provider.backends(n_qubits=7, operational=True, simulator=False))
# backend = provider.get_backend('ibm_oslo')
# print("Basis gates for:", backend)
# print(backend.configuration().basis_gates)
# print("Coupling map for:", backend)
# print(backend.configuration().coupling_map)
# instance = QuantumInstance(backend=backend, shots=shots)
# sampler = CircuitSampler(instance)
# expectation = PauliExpectation()


# Fix some global variables
n_qubits = 8 
tf = 4  # final time of the simulation
dt = 0.05  # time step
times = np.arange(0, tf+dt, dt)  # range of times for the simulation

# times = times[0::4]
# dt = times[1]-times[0]

## Model parameters

# 2D Hubbard model
model_params = [1,0.8]  # hopping and Coulomb parameters
lx, ly = 2, 1  # size of the square grid

# # XYZ Floquet model
# model_params = [1,0.8,0.6,0]

## Define the initial state
psi0 = QuantumCircuit(n_qubits)

# # 1D XYZ model + drive antiferromagnetic state
# flip_idx = list(range(n_qubits))[1::2]
# psi0.x(flip_idx)
# psi0.barrier()

# 2D Hubbard model half-filled antiferromagnetic state
fermion_idx = [0,2,5,7]  # !! HARDCODED for a 2x2 (4x1) lattice !!
psi0.x(fermion_idx)
psi0.barrier()

## List of observables to measure at each time t of the simulation
observables = ListOp([num_op(n_qubits, 0) @ num_op(n_qubits, 4),
               num_op(n_qubits, 0) @ num_op(n_qubits, 2)])
# observables = ListOp([one_site_op(n_qubits, 0, X),
#             one_site_op(n_qubits, 0, Z),
#             nearest_neigh_op(n_qubits, 0, [X, X]),
#             nearest_neigh_op(n_qubits, 0, [Z, Z])])

# Simulate the time evolution of psi0 by looping over multiple times:
obs_values = []
evolved_states = []
for t in times:
    print(f'Time: {t}')
    # n = int(t/dt)
    n = 5
    print(f'Number of Trotter steps: {n}')

    # Get the Hamiltonian at time t 
    hamiltonian = hubbard_ladder_hamiltonian(size_x=lx, model_params=model_params, time=None)
    # hamiltonian = partial(xyz_floquet_hamiltonian, n_qubits=n_qubits, model_params=model_params)

    # Prepare the Hamiltonian at the given time t
    qc_trot = trotterized_hamiltonian(
        hamiltonian=hamiltonian,
        trotter_steps=n,
        final_time=t
    )

    # # For driven xxx model (Jx=Jy=Jz):
    # dt = t/trotter_steps
    # astz = AnsatzXYZFloquet(n_qubits, depth=0)
    # for ts in range(trotter_steps):
    #     astz.add_a_trotter_step()

    #     params_rxx_yy_zz = astz.circuit.parameters[:-n_qubits]
    #     params_rz = astz.circuit.parameters[-n_qubits:]

    #     qc_trot = astz.circuit.assign_parameters(dict(zip(params_rxx_yy_zz, np.full(len(params_rxx_yy_zz), dt))))
    #     qc_trot = qc_trot.assign_parameters(dict(zip(params_rz, [(-1)**k * np.sin(ts*dt) * dt for k in range(n_qubits)])))

    # # For Hubbard:
    # astz = AnsatzLadderHubbard(n_qubits, depth=1)

    # num_rxx_ryy = sum([astz.circuit.count_ops()[x] for x in ["rxx","ryy"]])
    # num_rzz_rz = sum([astz.circuit.count_ops()[x] for x in ["rzz","rz"]])
    # params_rxx_ryy = astz.circuit.parameters[0:num_rxx_ryy]
    # params_rzz_rz = astz.circuit.parameters[num_rxx_ryy:]

    # one_trot_step = astz.circuit.assign_parameters(dict(zip(params_rxx_ryy, np.full(num_rxx_ryy, model_params[0]*t/n))))
    # one_trot_step = one_trot_step.assign_parameters(dict(zip(params_rzz_rz, np.full(num_rzz_rz, model_params[1]*t/n))))

    # qc_trot = QuantumCircuit(n_qubits)

    # for _ in range(n):
    #     qc_trot = qc_trot.compose(one_trot_step, range(n_qubits)) 

    # Apply the trotterized Hamiltonian on the initial condition and perform measurements on the time evolved state
    evolved_circ_state = psi0.compose(qc_trot)  
    evolved_states.append(evolved_circ_state)
    expectation_values = StateFn(observables, is_measurement=True) @ StateFn(evolved_circ_state)
    expectation_values = expectation.convert(expectation_values)
    sampled_op = sampler.convert(expectation_values)

    if shots > 1:
        variance = expectation.compute_variance(sampled_op)
        error = np.real(np.sqrt(np.array(variance) / shots))
        obs_values.append([np.real(np.array(sampled_op.eval())), error])

    else:
        obs_values.append(np.real(np.array(sampled_op.eval())))

# Get the depth and the number of CNOTs in the Trotter circuit (to be used when plotting the results)
decomposed_circ = qc_trot.decompose(reps=3)
depth = decomposed_circ.depth()
cnots = decomposed_circ.count_ops()["cx"]
print(f'Final circuit:\n{decomposed_circ}')
print(f'Depth: {depth}')
print(f'CNOTS: {cnots}')


# Save the data in a file (!! CAREFUL, file name HARDCODED !!)

# # Hubbard model
# n0n4, n0n2 = np.array(obs_values).T
# np.savez(
#     'data_and_figures/2d_hubbard/test_trotter_evolution_n=5_hubbard_2x2_params=1_0.8_bc=open.npz',
#     times=times,
#     depth=depth,
#     cnots=cnots,
#     n0n4=n0n4,
#     n0n2=n0n2 
# )

# # XYZ Floquet model
# x0, z0, x0x1, z0z1 = np.array(obs_values).T
# np.savez(
#     'data_and_figures/driven_xyz/trotter_fixed_dt/trotter_dt=0.05_driven_xyz_n_qubits={}_params=1_0.8_0.6_0_bc=open.npz'.format(n_qubits),
#     times=times,
#     depth=depth,
#     cnots=cnots,
#     x0=x0,
#     z0=z0,
#     x0x1=x0x1,
#     z0z1=z0z1,
#     evolved_states=evolved_states
# )
