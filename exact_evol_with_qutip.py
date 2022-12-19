# Import the relevant libraries
import numpy as np
from hamiltonian import *
from qutip import tensor, basis, mesolve, Qobj


# Define some relevant variables
n_qubits = 8
tf = 2
times = np.linspace(0, tf, 200)

## Define the Hamiltonian

# # 2D Hubbard model
# h = hubbard_2D_hamiltonian(size_x=2, size_y=2, model_params=[1,0.8])
# h = Qobj(h.to_matrix())

# 1D XYZ Floquet model
h_xyz = xyz_hamiltonian(n_qubits, [1,1,1,0])
h_drive = floquet_term(n_qubits)
coeff = lambda time, args: np.sin(args['w'] * time)
h = [Qobj(h_xyz.to_matrix()),[Qobj(h_drive.to_matrix()), coeff]] 

## Define the initial state
up = basis(2,0)
down = basis(2,1)
psi0 = [up]*n_qubits

# # 2D Hubbard model half-filled antiferromagnetic state
# flip_idx = [0,3,4,7] # !! HARDCODED for a 2x2 square lattice !!

# 1D XYZ Floquet model antiferromagnetic state
flip_idx = list(range(n_qubits))[0::2]

for i in flip_idx:
    psi0[i] = down
psi0 = Qobj(tensor(psi0).full())

## List of observables to measure at each time t of the simulation
# observables = [Qobj((num_op(n_qubits, 0) @ num_op(n_qubits, 1)).to_matrix()),
#                Qobj((num_op(n_qubits, 0) @ num_op(n_qubits, 4)).to_matrix())]
observables=[Qobj(one_site_op(n_qubits, 0, X).to_matrix()),
             Qobj(one_site_op(n_qubits, 0, Z).to_matrix()),
             Qobj(nearest_neigh_op(n_qubits, 0, [X, X]).to_matrix()),
             Qobj(nearest_neigh_op(n_qubits, 0, [Z, Z]).to_matrix())]

## Master equation evolution
output = mesolve(h, psi0, times, e_ops=observables, args={'w':1})
times = output.times

## Save the data in a file

# # Hubbard model
# n0n1, n0n4 = output.expect
# np.savez('data_and_figures/hubbard/exact_hubbard_2x2_params=1_0.8_bc=open.npz',
#          times=times, n0n1=n0n1, n0n4=n0n4)

# XYZ Floquet Hamiltonian
x0, z0, x0x1, z0z1 = output.expect
np.savez('data_and_figures/xyz_floquet/exact_xyz_floquet_params=1_1_1_0_bc=open.npz',
         times=times, x0=x0, z0=z0, x0x1=x0x1, z0z1=z0z1)