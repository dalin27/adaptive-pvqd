# Import the relevant libraries
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector
from qiskit.opflow import StateFn, CircuitSampler
import numpy as np
from utils import *
import time
from qiskit.synthesis import LieTrotter, SuzukiTrotter
# from scipy.optimize import minimize
# from qiskit.algorithms.optimizers import ADAM


class AdaptivePVQD:
    """ Class implementing the Adaptive pVQD algorithm """

    def __init__(
            self,
            ansatz,
            maxiter,
            fidelity_tolerance,
            gradient_tolerance,
            expectation,
            quantum_instance,
            pool_type
    ):
        """
        Args:
            ansatz (Ansatz class): contains the initial conditions and initial ansatz (if provided)
            maxiter(int): maximum number of optimization steps
            fidelity_tolerance (float): fidelity tolerance (\varepsilon) to stop the minimization routine
            gradient_tolerance (float): gradient tolerance (\delta) to stop the minimization routine
            expectation (ExpectationBase): expectation converter to evaluate expectation values
            quantum_instance (QuantumInstance): quantum backend to evaluate the circuits
            pool_type (str): either 'local' or 'non-local' operator pool
        """
        self.ansatz = ansatz
        self.maxiter = maxiter
        self.fidelity_tolerance = fidelity_tolerance
        self.gradient_tolerance = gradient_tolerance
        self.expectation = expectation
        self.quantum_instance = quantum_instance
        self.pool_type = pool_type

        self.n_qubits = len(self.ansatz.circuit.qubits)
        self.sampler = CircuitSampler(quantum_instance)

    def get_loss(
            self,
            hamiltonian,
            circuit_ansatz,
            var_parameters,
            time_step
    ):
        """ Evaluate the infidelity given the current parameters and the circuit ansatz.

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current numerical values for variational parameters.
            time_step (float): small time step.

        Returns:
            A function to evaluate the infidelity and a function to evaluate the gradient
        """

        # Get the time evolved circuit
        trotterized_ansatz = circuit_ansatz.assign_parameters(var_parameters)
        evolution_gate = PauliEvolutionGate(hamiltonian, time=time_step, synthesis=SuzukiTrotter(order=2, reps=2))  
        trotterized_ansatz.append(evolution_gate, circuit_ansatz.qubits)

        # Get the shifted circuit in parameter space
        shift = ParameterVector("dw", circuit_ansatz.num_parameters)
        shifted_ansatz = circuit_ansatz.assign_parameters(var_parameters + shift)

        # Get the overlap/fidelity between the time evoled and the shifted circuits
        state = StateFn(shifted_ansatz & trotterized_ansatz.inverse())
        overlap = StateFn(projector_zero_local(self.n_qubits), is_measurement=True) @ state
        overlap = self.expectation.convert(overlap)

        def infidelity(shift_values: np.array):
            """ Returns the infidelity value given some numerical values for the small shift parameters. """

            # Create the dictionary of "parameter object: value"
            if isinstance(shift_values, list):
                shift_values = np.asarray(shift_values)
                value_dict = {x_i: shift_values[:, i].tolist() for i, x_i in enumerate(shift)}
            else:
                value_dict = dict(zip(shift, shift_values))

            sampled_overlap = self.sampler.convert(overlap, params=value_dict)
            return 1 - np.real(sampled_overlap.eval())

        def gradient(shift_values: np.array):
            """ Returns the gradient of the infidelity using the parameter-shift rule. """
            dim = shift_values.size

            plus_shifts = (shift_values + np.pi/2 * np.identity(dim)).tolist()
            minus_shifts = (shift_values - np.pi/2 * np.identity(dim)).tolist()

            evaluated = infidelity(plus_shifts + minus_shifts)
            return (evaluated[:dim] - evaluated[dim:])/2

        def gradient_last_comp(shift_values):
            """ Computes only the last entry of the gradient. """

            last_comp_unit_vec = np.zeros(len(shift_values))
            last_comp_unit_vec[-1] = 1

            plus_shift = shift_values + np.pi/2 * last_comp_unit_vec
            minus_shift = shift_values - np.pi/2 * last_comp_unit_vec

            return (infidelity(plus_shift) - infidelity(minus_shift))/2

        return infidelity, gradient, gradient_last_comp

    def adaptive_step(
            self,
            hamiltonian,
            var_parameters,
            time_step,
            shift_init_guess
    ):
        """ Add a gate (from an operator pool) to the current circuit ansatz. Choose the gate that
        maximizes the gradient  (to converge faster to the minimum of the infidelity).

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            var_parameters (np.array): current variational parameters.
            time_step (float): small time step.
            shift_init_guess (np.array): guess for the classical optimization.

        Returns:
            new_circuit_ansatz (QuantumCircuit): circuit with the added gate
        """

        print('--------------------------')
        print(f'Start of an adaptive step')

        # Create the operator pool
        operator_pool = self.ansatz.create_op_pool(self.pool_type, ['x','y','z'], ['xx','yy','zz']) 

        # Define the array where the gradients will be stored
        new_gradients = np.zeros(len(operator_pool))

        # Loop over all the trial operators in the pool
        for i, operator in enumerate(operator_pool):
        
            # Returns a trial circuit without modifying self.ansatz.circuit
            trial_circuit_ansatz = self.ansatz.add_ops([operator], update_count=False)

            # Update the variational parameters and the shift vector by appending zero(s)
            # (because each trial operator has a single parameter, only one zero is appended, i.e. num_new_params=1)
            num_new_params = trial_circuit_ansatz.num_parameters - len(var_parameters)
            new_var_parameters = np.append(var_parameters, np.full(num_new_params, 0))
            new_shift_init_guess = np.append(shift_init_guess, np.full(num_new_params, 0))

            # Get the gradient function for this trial circuit (only for the newly added parameter(s), here we assume
            # that there is only one new parameter)
            _,_, gradient_last_comp = self.get_loss(hamiltonian, trial_circuit_ansatz, new_var_parameters, time_step)
            
            # The only entries in the gradient vector that change are the ones corresponding to the
            # added variational parameters (again, here there is only one new parameter)
            new_gradients[i] = np.abs(gradient_last_comp(new_shift_init_guess))

        # Make a dictionary to pair each trial operator to its gradient
        operator_pool_keys = [op + '_' + str(pos) for op, pos in operator_pool]	
        ops_and_grad = dict(zip(operator_pool_keys, new_gradients))	
        ops_and_grad_cp = dict(ops_and_grad)  # make a copy of the dictionary
        
        ## Method 1: Tetris-like -- loop until all the qubit indices have been exhausted (i.e. until the pool is empty)
        operators = []  # to store the operators to add to the current parametrized circuit
        while len(ops_and_grad_cp) > 0:
            # Select the operator in the current pool that maximizes the gradient
            op_max_grad = max(ops_and_grad_cp, key=ops_and_grad_cp.get).split('_')
            op_max_grad[1] = eval(op_max_grad[1])
            operators.append(tuple(op_max_grad))
            
            # Remove all operators in the pool that act on qubit indices already acted on by op_max_grad
            [ops_and_grad_cp.pop(op) for op in ops_and_grad_cp.copy().keys() if set(eval(op.split('_')[1])).isdisjoint(op_max_grad[1]) == False]

        # ## Method 2: Add simply one gate per adaptive step
        # # Select the operator in the current pool that maximizes the gradient
        # op_max_grad = max(ops_and_grad_cp, key=ops_and_grad_cp.get).split('_')
        # op_max_grad[1] = eval(op_max_grad[1])
        # operators = [tuple(op_max_grad)]

        # Sort the operators according to the qubit(s) on which they act and add them to the circuit
        sorted_idx = np.argsort([op[1][0] for op in operators])
        operators = [operators[i] for i in sorted_idx]
        new_circuit_ansatz = self.ansatz.add_ops(operators, update_count=True)
        print(f'Operators added to the circuit: {operators}')
        # print(f'New circuit ansatz: \n{new_circuit_ansatz}')

        # ## Method 3: Add a Trotter step/block to the parametrized circuit
        # self.ansatz.add_a_trotter_step()
        # print(f"A Trotter step has been added to the circuit ansatz.")
        # print(self.ansatz.circuit)
        # new_circuit_ansatz = self.ansatz.circuit

        self.ansatz.circuit = new_circuit_ansatz  # Overwrite the current circuit
        return new_circuit_ansatz

    def minimization_routine(
            self,
            var_parameters,
            shift_init_guess,
            hamiltonian,
            circuit_ansatz,
            time_step
    ):
        """ Find the small parameters dw* that minimize the infidelity.

        Args:
            var_parameters (np.array): current variational parameters.
            shift_init_guess (np.array): guess for the classical optimization
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            time_step (float): small time step.

        Returns:
            approximate minimum of the infidelity
        """

        print('-------------------------------')
        print('Start optimizing the infidelity')

        # Get the infidelity and gradient functions
        infidelity, gradient,_ = self.get_loss(hamiltonian, circuit_ansatz, var_parameters, time_step)


        ## Homemade ADAM
        updated_shift = np.copy(shift_init_guess)
        fid = 1 - infidelity(updated_shift)
        grad_norm = max(np.abs(gradient(updated_shift)))  # infinite norm

        # Initialize the 1st and 2nd moment vectors of Adam
        m = np.zeros(len(var_parameters))
        v = np.zeros(len(var_parameters))

        count = 0  # to count the number of iterations of Adam
        while grad_norm > self.gradient_tolerance and count < self.maxiter:  
            count += 1

            grad = gradient(updated_shift)
            grad_norm = max(np.abs(grad))

            updated_shift, m, v = adam_step(updated_shift, count, m, v, grad)

            # Get the optimized fidelity
            fid = 1 - infidelity(updated_shift)
            
            # Print statement
            if count == 1 or count % 25 == 0:
                print(f'Iteration: {count} Fidelity: {fid:.8f} Gradient: {grad_norm:.8f}')


        ## Scipy Minimize
        # self.opt_iter = 0
        # def callback_func(x):
        #     fid = 1-infidelity(x)
        #     if self.opt_iter % 20 == 0:
        #         print(f'Iteration: {self.opt_iter}   Fidelity: {fid}')
        #     self.opt_iter += 1
        #     # if fid > fidelity_threshold:
        #     #     raise Trigger

        # optimizer_result = minimize(
        #     fun=infidelity,
        #     x0=shift_init_guess,
        #     jac=gradient,
        #     # method='CG',
        #     callback=callback_func,
        #     # options={'maxiter': self.maxiter}
        # )
        # updated_shift = optimizer_result.x
        # fid = 1 - optimizer_result.fun


        ## Qiskit Minimize
        # optimizer_result = ADAM(maxiter=500, amsgrad=True).minimize(
        #     fun=infidelity,
        #     x0=shift_init_guess,
        #     jac=gradient,
        # )
        # updated_shift = optimizer_result.x
        # fid = 1 - optimizer_result.fun
        # print(f'Current fidelity: {fid:.8f}')


        return updated_shift, fid

    def one_time_step(
            self,
            hamiltonian,
            circuit_ansatz,
            var_parameters,
            time_step,
            shift_init_guess,
    ):
        """ Advance the time evolution by one time step dt in time.

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current variational parameters.
            time_step (float): small time step.
            shift_init_guess (np.array): guess for the classical optimization

        Returns:
            A tuple with the new variational parameters and the fidelity
        """

        if circuit_ansatz.num_parameters == 0:
            # When there are no parameters in the circuit (at t=0 when no ansatz is provided)  
            updated_shift = np.copy(shift_init_guess)
            fidelity = 0  # set the fidelity to any arbitrary value below "self.fidelity_tolerance"
        else: 
            # For t>0 (when there is at least one parameter in the circuit).
            # Find the current best small shift parameters dw* that minimize the infidelity
            updated_shift, fidelity = self.minimization_routine(
                var_parameters,
                shift_init_guess,
                hamiltonian,
                circuit_ansatz,
                time_step
            )

        # Adaptive part (comment out the while loop if adaptive part should not be used):
        updated_var_parameters = np.copy(var_parameters)
        while fidelity < self.fidelity_tolerance:
            
            # Perform and adaptive step
            new_circuit_ansatz = self.adaptive_step(hamiltonian, updated_var_parameters, time_step, updated_shift)
    
            # Update the parameters w and the small shift parameters dw by appending zero(s)
            total_num_new_params = new_circuit_ansatz.num_parameters - len(var_parameters)
            num_new_params = new_circuit_ansatz.num_parameters - len(updated_var_parameters)
            updated_var_parameters = np.append(var_parameters, np.full(total_num_new_params, 0))
            updated_shift = np.append(updated_shift, np.full(num_new_params, 0))
    
            # Find the current best small shift parameters dw* that minimize the infidelity
            updated_shift, fidelity = self.minimization_routine(
                updated_var_parameters,
                updated_shift,
                hamiltonian,
                new_circuit_ansatz,
                time_step
            )

        return updated_var_parameters + updated_shift, fidelity

    def evolve(
            self,
            hamiltonian,
            num_time_steps,
            initial_time,
            final_time,
            initial_parameters,
            shift_init_guess,
            observables
    ):
        """ Main function performing the actual time evolution.

        Args:
            hamiltonian (functools.partial): Hamiltonian used for the time evolution.
            num_time_steps (int): number of time steps for the time evolution.
            initial_time (float): initial time of the evolution (non-zero if pre-saved data available)
            final_time (float): final time of the evolution
            initial_parameters (np.array): initial parameters of the circuit ansatz.
            shift_init_guess (np.array): guess for the classical optimization of the infidelity
            observables (List[PauliOp]): list of Pauli strings to evaluate

        Returns:
            data_log (dict): contains all the information/results of the algorithm.
        """

        start_time = time.time()
        print(f'Hamiltonian:\n{hamiltonian(time=initial_time)}')
        print(f'Initial circuit:\n{self.ansatz.circuit}')

        # Get the function to evaluate the expectation value of the observables of interest given
        # the current parametrized circuit
        evaluate_observables = get_observable_evaluator(self.ansatz.circuit, observables, self.expectation, self.sampler)

        # Time related stuff
        time_step = (final_time - initial_time) / num_time_steps
        times = np.linspace(initial_time, final_time, num_time_steps + 1).tolist()  # +1 because we include ti and tf
        
        # Initialize the data log at t=0
        data_log = {"times": times,
                    "parameters": [initial_parameters],
                    "fidelity": [1],
                    "observables values": [evaluate_observables(initial_parameters)],
                    "evolved state": []
                    }
        
        evolved_state = self.ansatz.circuit.bind_parameters(initial_parameters)
        data_log["evolved state"].append(evolved_state)

        # Loop over all time steps (to reach the final time)
        for tt in times[1:]:
            
            print(f"Time step: {tt:<10.3f}")
            ham = hamiltonian(time=tt)

            # Perform one time step
            next_parameters, fidelity = self.one_time_step(
                ham,
                self.ansatz.circuit,
                data_log["parameters"][-1],
                time_step,
                shift_init_guess
            )
            shift_init_guess = np.zeros(len(next_parameters))

            print(f"Fidelity: {fidelity}")
            print(f"Elapsed time since beginning: {time_convert(time.time() - start_time)}")
            print('==================================================')

            # Get the function to evaluate the expectation value of the observables of interest given
            # the current parametrized circuit
            evaluate_observables = get_observable_evaluator(self.ansatz.circuit, observables, self.expectation, self.sampler)

            # Compute and store the relevant quantities
            data_log["parameters"].append(next_parameters)
            data_log["fidelity"].append(fidelity)
            data_log["observables values"].append(evaluate_observables(next_parameters))
            evolved_state = self.ansatz.circuit.bind_parameters(next_parameters)
            data_log["evolved state"].append(evolved_state)

        data_log["final circuit"] = [self.ansatz.circuit]

        return data_log
