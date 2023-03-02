# Import the relevant libraries
from qiskit.opflow import ListOp, StateFn
import numpy as np
from hamiltonian import I, Z, one_site_op


def adam_step(theta, t, m, v, grad):
    """ One step of the Adam optimizer, following the structure and notation suggested in
     https://arxiv.org/abs/1412.6980.

    Args:
        theta (np.array): parameters to be optimized
        t (int): time step
        m (np.array): 1st moment vector
        v (np.array): 2nd moment vector
        grad (np.array): gradient of the objective function to be optimized at time step t

    Returns:
        new_theta (np.array): updated parameters
    """

    # Default settings (except for alpha) suggested in the paper:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    alpha = 0.005

    # Implement one step of the Adam optimizer
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2
    alphat = alpha * np.sqrt(1-beta2**t)/(1-beta1**t)
    new_theta = theta - alphat * m/(np.sqrt(v) + eps)

    return new_theta, m, v


def get_observable_evaluator(
        ansatz,
        observables,
        expectation,
        sampler
):
    """ Compute the expectation value of a list of observables given a parametrized circuit.

    Args:
        ansatz (QuantumCircuit): parametrized/variational circuit.
        observables (List[PauliOp]): list of Pauli strings to evaluate.
        expectation (ExpectationBase): expectation converter to evaluate expectation values.
        sampler (CircuitSampler): uses a backend to convert any StateFn into an approximation of the
                                  state function.

    Returns:
        A function that computes the expectation value of a list of operators.
    """
    if isinstance(observables, list):
        observables = ListOp(observables)

    expectation_values = StateFn(observables, is_measurement=True) @ StateFn(ansatz)
    expectation_values = expectation.convert(expectation_values)

    ansatz_parameters = ansatz.parameters

    def evaluate_observables(parameter_values: np.array):
        """ Evaluate the observables given the numerical parameters provided. """
        if len(parameter_values) != len(ansatz_parameters):
            raise (f"The number of parameter values provided does not match the number of free "
                   f"parameters in the circuit."
                   )

        value_dict = dict(zip(ansatz_parameters, parameter_values))
        sampled_op = sampler.convert(expectation_values, params=value_dict)
        return np.real(np.array(sampled_op.eval()))

    return evaluate_observables


# def projector_zero_global(num_qubits):
#     return tens_prod([0.5 * (I + Z)] * num_qubits)


def projector_zero_local(num_qubits):
    tot_prj = 0  # initialize the sum of all local projectors

    for k in range(num_qubits):
        tot_prj += one_site_op(num_qubits, k, 0.5 * (I + Z))

    return tot_prj / num_qubits  # returns a PauliSumOp object


def time_convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return '%d:%02d:%02d' % (hour, min, sec)