# Import the relevant libraries
from qiskit import QuantumCircuit
from hamiltonian import *
from qiskit.circuit import Parameter
from itertools import combinations


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
        """ Generate parameter objects with names that can be sorted alphabetically (this
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

            # To sort the parameters alphabetically, generate the appropriate amount of zeros.
            # If more than 1000 parameters are expected, add another range from 1000 to 9999, etc.
            param_ranges = [[0, 9], [10, 99],[100,999]]
            for k, param_range in enumerate(param_ranges):
                if param_range[0] <= count <= param_range[1]:
                    str_zeros = ('0' * len(param_ranges))[k:]
                    break
                else:
                    continue
            try: 
                # If "str_zeros" has been defined, return it
                str_zeros
            except NameError:
                # If "str_zeros" has not been defined, print an error message
                print("Another range from 1000 to 9999 must be added to param_ranges.")

            params.append(Parameter(str_zeros + str(count)))
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
        idx_pairs = list(combinations(range(n), 2))  # n choose 2 index pairs

        pool += list(map(lambda op: [(op, [i]) for i in range(n)], one_body_ops_labels))
        pool += list(map(lambda op: [(op, list(idx)) for idx in idx_pairs], two_body_ops_labels))

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
            if op == 'cu':
                p = self.create_params(3, update_count)
                circuit.cu3(*p, *pos)
        circuit.barrier()

        return circuit


class AnsatzHubbard2D(Ansatz):
    """ Circuit ansatz for the 2D Hubbard model (only contains initial conditions). """

    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(n_qubits)

        # Define the initial state to be in a half-filled antiferromagnetic state
        fermion_idx = [0,3,4,7]  # !! HARDCODED for a 2x2 square lattice !!
        self.circuit.x(fermion_idx)
        self.circuit.barrier()


class AnsatzXYZFloquet(Ansatz):
    """ Circuit ansatz for the 1D XYZ Floquet model (only contains initial conditions). """

    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(n_qubits)
        
        # Define the initial state to be in an antiferromagnetic state
        odd_idx = list(range(n_qubits))[1::2]
        self.circuit.x(odd_idx)
        self.circuit.barrier()


class AnsatzRxRxxRyRyyRzRzz(Ansatz):
    """ (Fixed) Circuit ansatz where blocks of the form Rx-Rxx-Ry-Ryy-Rz-Rzz are appended after 
    the initial conditions. The number of blocks appended is specified by the "depth" parameter.
    """

    def __init__(self, n_qubits, depth):
        """
        Args:
            n_qubits (int): number of qubits
            depth (int): depth of the initial circuit
        """
        super().__init__(n_qubits)

        # Define the initial state to be in a half-filled antiferromagnetic state
        fermion_idx = [0,3,4,7]  # !! HARDCODED for a 2x2 square lattice !!
        self.circuit.x(fermion_idx)
        self.circuit.barrier()

        # Rx-Rxx-Ry-Ryy-Rz-Rzz blocks
        for _ in range(depth):
            self.add_a_block()

    def add_a_block(self):

        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.rx(*p, i)
        self.circuit.barrier()

        for i in range(self.n_qubits - 1):
            p = self.create_params(1, update_count=True)
            self.circuit.rxx(*p, i, (i + 1) % self.n_qubits)
        self.circuit.barrier()

        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.ry(*p, i)
        self.circuit.barrier()

        for i in range(self.n_qubits - 1):
            p = self.create_params(1, update_count=True)
            self.circuit.ryy(*p, i, (i + 1) % self.n_qubits)
        self.circuit.barrier()

        for i in range(self.n_qubits):
            p = self.create_params(1, update_count=True)
            self.circuit.rz(*p, i)
        self.circuit.barrier()
        
        for i in range(self.n_qubits - 1):
            p = self.create_params(1, update_count=True)
            self.circuit.rzz(*p, i, (i + 1) % self.n_qubits)
        self.circuit.barrier()

