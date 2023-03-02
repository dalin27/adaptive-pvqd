# Import the relevant libraries
from qiskit import QuantumCircuit
from hamiltonian import *
from qiskit.circuit import Parameter
from itertools import combinations
import qiskit.quantum_info as qi
from qiskit.circuit.library import XXPlusYYGate

class Ansatz:
    """ Base class for the circuit ansatz. """

    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): number of qubits
        """
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.count = 0  # count the number of operators/parameters in the circuit

    def create_params(self, num_params, update_count):
        """ Generate parameter objects with labels that can be sorted alphabetically (this
        requirement comes from the "parameters" method of the QuantumCircuit class - see the
        documentation of the Qiskit source code for details).

        Args:
            num_params (int): number of parameter objects to be created
            update_count (bool): specify if the total count number should be globally updated

        Returns:
            params (List[ParameterExpression,...]): the generated parameter objects
        """
        count = self.count
        params = []

        for _ in range(num_params):

            # Append zeros to sort the parameter labels alphabetically
            if count < 1000:
                label_count = str(count).zfill(4)
            else:
                raise Exception('Change the line of code above to [...].zfill(n), n>4.')

            params.append(Parameter(label_count))
            count += 1

        if update_count:
            self.count = count

        return params

    def create_op_pool(self, one_body_ops_labels, two_body_ops_labels):
        """ Create the operator pool.
        Args:
            one_body_ops_labels (List[str,...]): e.g. ['x','y','z']
            two_body_ops_labels (List[str,...]): e.g. ['xx','yy','zz']

        Returns:
            pool (List[tuple[str,list]]): operators and their position
        """
        n = self.n_qubits
        pool = []  # initialize a list to store all the trial operators 
        # idx_pairs = list(combinations(range(n), 2))  # n choose 2 index pairs

        pool += list(map(lambda op: [(op, [i]) for i in range(n)], one_body_ops_labels))
        pool += list(map(lambda op: [(op, [i, (i+1) % n]) for i in range(n-1)], two_body_ops_labels))
        # pool += list(map(lambda op: [(op, list(idx)) for idx in idx_pairs], two_body_ops_labels))

        # Flatten the pool
        pool = [tup for sublist in pool for tup in sublist]

        return pool

    def add_ops(self, operators, update_count):
        """ Add (an) operator(s) to the circuit ansatz.

        Args:
            operators (List[tuple[str,list]): operators (in the pool) and their positions
            update_count (bool): specify if the total count number should be globally updated

        Returns:
            circuit (QuantumCircuit): previous circuit ansatz with (an) added operator(s)
        """
        circuit = self.circuit.copy()  # to avoid overwriting

        for op, pos in operators:

            if op == 'x':
                p = self.create_params(1, update_count)
                circuit.rx(*p, *pos)
            if op == 'y':
                p = self.create_params(1, update_count)
                circuit.ry(*p, *pos)
            if op == 'z':
                p = self.create_params(1, update_count)
                circuit.rz(*p, *pos)
            if op == 'u':
                p = self.create_params(3, update_count)
                circuit.u(*p, *pos)
            if op == 'xx':
                p = self.create_params(1, update_count)
                circuit.rxx(*p, *pos)
            if op == 'yy':
                p = self.create_params(1, update_count)
                circuit.ryy(*p, *pos)
            if op == 'zz':
                p = self.create_params(1, update_count)
                circuit.rzz(*p, *pos)
            if op == 'zx':
                p = self.create_params(1, update_count)
                circuit.rzx(*p, *pos)
        circuit.barrier()

        return circuit


class AnsatzHubbard1D(Ansatz):
    """ Circuit ansatz for the 1D Hubbard model. 
	Fermionic index ordering: all spin up then all spin down modes. """

    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(n_qubits)

        # Define the initial state to be in a half-filled antiferromagnetic state
        indices_flip_qubit = list(range(0, int(self.n_qubits/2), 2)) + list(range(int(self.n_qubits/2+1), n_qubits, 2))  
        self.circuit.x(indices_flip_qubit)
        self.circuit.barrier()

    def add_a_trotter_step(self):
        
        # Hopping terms
        hop_indices = list(range(int(self.n_qubits/2-1)))
        hop_indices = hop_indices[0::2] + hop_indices[1::2]
        hop_indices += list(np.array(hop_indices)+int(self.n_qubits/2))

        for i in hop_indices:
            p = self.create_params(1, update_count=True)
            self.circuit.rxx(*p, i, i+1) 
            p = self.create_params(1, update_count=True)
            self.circuit.ryy(*p, i, i+1) 
            # self.circuit.append(XXPlusYYGate(*p, beta=0), [i, i+1])
        self.circuit.barrier()

        # On-site interaction terms
        int_indices = list(range(self.n_qubits))
        int_index_pairs = zip(int_indices[0:int(self.n_qubits/2)], int_indices[int(self.n_qubits/2):])

        for i, j in int_index_pairs:
            p = self.create_params(1, update_count=True)
            # self.circuit.cp(*p, i, j)
            self.circuit.rzz(*p, i, j)
        self.circuit.barrier()
        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.rz(*p, i)
        self.circuit.barrier()


class AnsatzLadderHubbard(Ansatz):
    """ Circuit ansatz for the Hubbard model on a Lx x 2 lattice. """

    def __init__(self, n_qubits, depth):
        """
        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(n_qubits)

        # Define the initial state to be in a half-filled antiferromagnetic state
        indices_flip_qubit = list(range(0, int(n_qubits/2), 2)) + list(range(int(n_qubits/2+1), n_qubits, 2))  
        self.circuit.x(indices_flip_qubit)
        self.circuit.barrier()

        # Implement the fermionic swap gate
        self.fswap = qi.Operator([[1,0,0,0],
                                  [0,0,1,0],
                                  [0,1,0,0],
                                  [0,0,0,-1]])

        # For a fixed Trotter ansatz:
        for _ in range(depth):
            self.add_a_trotter_step()

    def add_a_trotter_step(self):

        # Hopping terms
        hop_indices = list(range(int(self.n_qubits/2-1)))
        hop_indices = hop_indices[0::2] + hop_indices[1::2]
        hop_indices += list(np.array(hop_indices)+int(self.n_qubits/2))

        for i in hop_indices:
            p = self.create_params(1, update_count=True)
            self.circuit.rxx(*p, i, i+1) 
            p = self.create_params(1, update_count=True)
            self.circuit.ryy(*p, i, i+1) 
            # self.circuit.append(XXPlusYYGate(*p, beta=0), [i, i+1])
        self.circuit.barrier()

        fswap_indices = [[i,i+1] for i in range(0, self.n_qubits, 2)]
        [self.circuit.unitary(self.fswap, [i,j], label='fswap') for i, j in fswap_indices]

        # Hopping terms
        hop_indices = list(range(1, int(self.n_qubits/2-1), 2))
        hop_indices += list(np.array(hop_indices)+int(self.n_qubits/2))

        for i in hop_indices:
            p = self.create_params(1, update_count=True)
            self.circuit.rxx(*p, i, i+1) 
            p = self.create_params(1, update_count=True)
            self.circuit.ryy(*p, i, i+1) 
            # self.circuit.append(XXPlusYYGate(*p, beta=0), [i, i+1])
        self.circuit.barrier()

        fswap_indices = [[i,i+1] for i in range(0, self.n_qubits, 2)]
        [self.circuit.unitary(self.fswap, [i,j], label='fswap') for i, j in fswap_indices]

        # On-site interaction terms
        int_indices = list(range(self.n_qubits))
        int_index_pairs = zip(int_indices[0:int(self.n_qubits/2)], int_indices[int(self.n_qubits/2):])

        for i, j in int_index_pairs:
            p = self.create_params(1, update_count=True)
            self.circuit.rzz(*p, i, j)
            # self.circuit.cp(*p, i, j)
        self.circuit.barrier()
        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.rz(*p, i)
        self.circuit.barrier()


class AnsatzXYZFloquet(Ansatz):
    """ Circuit ansatz for the 1D XYZ Floquet model. """

    def __init__(self, n_qubits, depth):
        """
        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(n_qubits)
        
        # Define the initial state to be in an antiferromagnetic state
        odd_idx = list(range(n_qubits))[1::2]
        self.circuit.x(odd_idx)
        self.circuit.barrier()

        # For a fixed Trotter ansatz:
        for _ in range(depth):
            self.add_a_trotter_step()

    def add_a_trotter_step(self):
        for j in [0,1]:
            for i in range(j, self.n_qubits-1, 2):
                p = self.create_params(1, update_count=True)
                self.circuit.rxx(*p, i, i+1)
                p = self.create_params(1, update_count=True)
                self.circuit.ryy(*p, i, i+1)
                p = self.create_params(1, update_count=True)
                self.circuit.rzz(*p, i, i+1)
        self.circuit.barrier()
        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.rz(*p, i)
        self.circuit.barrier()

    #     # Implement Rxx-yy-zz with 3 CNOTS instead of 6 (only when Jx=Jy=Jz)
    #     for j in [0,1]:
    #         for i in range(j, self.n_qubits-1, 2):
    #             p = self.create_params(1, update_count=True)
    #             self.apply_rxx_yy_zz(*p,i,i+1)
    #     self.circuit.barrier()
    #     for i in range(self.n_qubits):
    #         p = self.create_params(1, update_count=True)
    #         self.circuit.rz(*p, i)
    #     self.circuit.barrier()

    # def apply_rxx_yy_zz(self, delta, i, j):
    #     # Implementation in Tacchino et al. (2020)
    #     self.circuit.cnot(i,j)
    #     self.circuit.rx(2*delta-np.pi/2,i)
    #     self.circuit.rz(2*delta,j)
    #     self.circuit.h(i)
    #     self.circuit.cnot(i,j)
    #     self.circuit.h(i)
    #     self.circuit.rz(-2*delta,j)
    #     self.circuit.cnot(i,j)
    #     self.circuit.rx(np.pi/2,i)
    #     self.circuit.rx(-np.pi/2,j)
