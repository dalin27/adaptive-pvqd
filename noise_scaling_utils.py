## Function to create a modified error map
import logging
from warnings import warn

import numpy as np
from numpy import inf, exp, allclose

import qiskit.quantum_info as qi
from qiskit.circuit import Gate, Measure, Instruction, Delay
from qiskit.circuit.library import IGate,XGate,YGate, ZGate, Reset
from qiskit.quantum_info.operators.channel import Choi, Kraus

from qiskit_aer.noise.device.parameters      import _NANOSECOND_UNITS
from qiskit_aer.noise.device.parameters      import gate_param_values
from qiskit_aer.noise.device.parameters      import readout_error_values
from qiskit_aer.noise.device.parameters      import thermal_relaxation_values
from qiskit_aer.noise.errors.readout_error   import ReadoutError
from qiskit_aer.noise.errors.standard_errors import depolarizing_error, QuantumError
from qiskit_aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit_aer.noise.noiseerror             import NoiseError
from qiskit_aer.noise.passes                 import RelaxationNoisePass



from qiskit.providers            import QubitProperties
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.providers.models     import BackendProperties
from qiskit.providers.aer.noise  import NoiseModel



## This is a modified version of the noise scaling utils from Qiskit Aer

# -----------------------------------------------------------------------------
## Utils functions

def _truncate_t2_value(t1, t2):
    """Return t2 value truncated to 2 * t1 (for t2 > 2 * t1)"""
    if t1 is None:
        return t2
    elif t2 is None:
        return 2 * t1
    return min(t2, 2 * t1)


def _excited_population(freq, temperature):
    """Return excited state population from freq [Hz] and temperature [mK]."""
    if freq is None or temperature is None:
        return 0
    population = 0
    if freq != inf and temperature != 0:
        # Compute the excited state population from qubit frequency and temperature
        # based on Maxwell-Boltzmann distribution
        # considering only qubit states (|0> and |1>), i.e. truncating higher energy states.
        # Boltzman constant  kB = 8.617333262e-5 (eV/K)
        # Planck constant h = 4.135667696e-15 (eV.s)
        # qubit temperature temperatue = T (mK)
        # qubit frequency frequency = f (Hz)
        # excited state population = 1/(1+exp((h*f)/(kb*T*1e-3)))
        # See e.g. Phys. Rev. Lett. 114, 240501 (2015).
        exp_param = exp((47.99243 * 1e-9 * freq) / abs(temperature))
        population = 1 / (1 + exp_param)
        if temperature < 0:
            # negative temperate implies |1> is thermal ground
            population = 1 - population
    return population


def _combine_depol_and_relax_error(depol_error, relax_error):
    if depol_error and relax_error:
        return depol_error.compose(relax_error)
    if depol_error:
        return depol_error
    if relax_error:
        return relax_error
    return None


def rescale_basic_device_target_gate_errors(
    target, gate_error=True, thermal_relaxation=True, temperature=0, scale=1
):
    """Return QuantumErrors derived from a devices Target.
    Note that, in the resulting error list, non-Gate instructions (e.g. Reset) will have
    no gate errors while they may have thermal relaxation errors. Exceptionally,
    Measure instruction will have no errors, neither gate errors nor relaxation errors.

    Note: Units in use: Time [s], Frequency [Hz], Temperature [mK]
    """
    errors = []
    for op_name, inst_prop_dic in target.items():
        operation = target.operation_from_name(op_name)
        if isinstance(operation, Measure):
            continue
        if inst_prop_dic is None:  # ideal simulator
            continue
        for qubits, inst_prop in inst_prop_dic.items():
            if inst_prop is None:
                continue
            depol_error = None
            relax_error = None
            # Get relaxation error
            if thermal_relaxation and inst_prop.duration:
                relax_params = {
                    q: (
                        target.qubit_properties[q].t1,
                        target.qubit_properties[q].t2,
                        target.qubit_properties[q].frequency,
                    )
                    for q in qubits
                }
                relax_error = rescale_device_thermal_relaxation_error(
                    qubits=qubits,
                    gate_time=inst_prop.duration,
                    relax_params=relax_params,
                    temperature=temperature,
                    scale=scale
                )
            # Get depolarizing error
            if gate_error and inst_prop.error and isinstance(operation, Gate):
                depol_error = rescale_device_depolarizing_error(
                    qubits=qubits,
                    error_param=inst_prop.error,
                    relax_error=relax_error,
                    scale=scale
                )
            # Combine errors
            combined_error = _combine_depol_and_relax_error(depol_error, relax_error)
            if combined_error:
                errors.append((op_name, qubits, combined_error))

    return errors


# -----------------------------------------------------------------------------
# Functions to generate the different rescaled errors from device properties

def rescale_device_depolarizing_error(qubits, error_param, relax_error=None,scale=1):
    """Construct a depolarizing_error for device.
    If un-physical parameters are supplied, they are truncated to the theoretical bound values."""

    # We now deduce the depolarizing channel error parameter in the
    # presence of T1/T2 thermal relaxation. We assume the gate error
    # parameter is given by e = 1 - F where F is the average gate fidelity,
    # and that this average gate fidelity is for the composition
    # of a T1/T2 thermal relaxation channel and a depolarizing channel.

    # For the n-qubit depolarizing channel E_dep = (1-p) * I + p * D, where
    # I is the identity channel and D is the completely depolarizing
    # channel. To compose the errors we solve for the equation
    # F = F(E_dep * E_relax)
    #   = (1 - p) * F(I * E_relax) + p * F(D * E_relax)
    #   = (1 - p) * F(E_relax) + p * F(D)
    #   = F(E_relax) - p * (dim * F(E_relax) - 1) / dim

    # Hence we have that the depolarizing error probability
    # for the composed depolarization channel is
    # p = dim * (F(E_relax) - F) / (dim * F(E_relax) - 1)
    if relax_error is not None:
        relax_fid = qi.average_gate_fidelity(relax_error)
        relax_infid = 1 - relax_fid
    else:
        relax_fid = 1
        relax_infid = 0
    if error_param is not None and error_param > relax_infid:
        num_qubits = len(qubits)
        dim = 2**num_qubits
        error_max = dim / (dim + 1)
        # Check if reported error param is un-physical
        # The minimum average gate fidelity is F_min = 1 / (dim + 1)
        # So the maximum gate error is 1 - F_min = dim / (dim + 1)
        error_param = min(error_param, error_max)
        # Model gate error entirely as depolarizing error
        num_qubits = len(qubits)
        dim = 2**num_qubits
        depol_param = dim * (error_param - relax_infid) / (dim * relax_fid - 1)
        max_param = 4**num_qubits / (4**num_qubits - 1)
        if depol_param > max_param:
            depol_param = min(depol_param, max_param)
        return depolarizing_error(scale*depol_param, num_qubits)
    return None


def rescale_device_thermal_relaxation_error(qubits, gate_time, relax_params, temperature,scale=1):
    """Construct a thermal_relaxation_error for device.

    Expected units: frequency in relax_params [Hz], temperature [mK].
    Note that gate_time and T1/T2 in relax_params must be in the same time unit.
    """
    # Check trivial case
    if gate_time is None or gate_time == 0:
        return None

    # Construct a tensor product of single qubit relaxation errors
    # for any multi qubit gates
    first = True
    error = None
    for qubit in qubits:
        t1, t2, freq = relax_params[qubit]
        t2 = _truncate_t2_value(t1, t2)
        if t1 is None:
            t1 = inf
        if t2 is None:
            t2 = inf
        population = _excited_population(freq, temperature)
        if first:
            error = new_rescale_thermal_relaxation_error(t1, t2, gate_time, population,scale)
            first = False
        else:
            single = new_rescale_thermal_relaxation_error(t1, t2, gate_time, population,scale)
            error = error.expand(single)
    return error

def rescale_thermal_relaxation_error(t1, t2, time, excited_state_population=0,scale=1):
    r"""
    Return a single-qubit thermal relaxation quantum error channel.

    Args:
        t1 (double): the :math:`T_1` relaxation time constant.
        t2 (double): the :math:`T_2` relaxation time constant.
        time (double): the gate time for relaxation error.
        excited_state_population (double): the population of :math:`|1\rangle`
                                           state at equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Raises:
        NoiseError: If noise parameters are invalid.

    Additional information:
        * For parameters to be valid :math:`T_1` and :math:`T_2` must
          satisfy :math:`T_2 \le 2 T_1`.

        * If :math:`T_2 \le T_1` the error can be expressed as a mixed
          reset and unitary error channel.

        * If :math:`T_1 < T_2 \le 2 T_1` the error must be expressed as a
          general non-unitary Kraus error channel.
    """
    if excited_state_population < 0:
        raise NoiseError(
            "Invalid excited state population " "({} < 0).".format(excited_state_population)
        )
    if excited_state_population > 1:
        raise NoiseError(
            "Invalid excited state population " "({} > 1).".format(excited_state_population)
        )
    if time < 0:
        raise NoiseError("Invalid gate_time ({} < 0)".format(time))
    if t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 - 2 * t1 > 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # T1 relaxation rate
    if t1 == np.inf:
        rate1 = 0
        p_reset = 0
    else:
        rate1 = 1 / t1
        p_reset = 1 - np.exp(-time * rate1)
    # T2 dephasing rate
    if t2 == np.inf:
        rate2 = 0
        exp_t2 = 1
    else:
        rate2 = 1 / t2
        exp_t2 = np.exp(-time * rate2)
    # Qubit state equilibrium probabilities
    p0 = 1 - excited_state_population
    p1 = excited_state_population

    if t2 > t1:
        # If T_2 > T_1 we must express this as a Kraus channel
        # We start with the Choi-matrix representation:
        if scale >0:
            chan = Choi(
                np.array(
                    [
                        [1 - scale*p1 * p_reset, 0, 0, scale*exp_t2],
                        [0, scale*p1 * p_reset, 0, 0],
                        [0, 0, scale*p0 * p_reset, 0],
                        [scale*exp_t2, 0, 0, 1 - scale*p0 * p_reset],
                    ]
                )
            )
            return QuantumError(Kraus(chan))
        elif scale == 0:
            circuits = [[(IGate(), [0])]]
            probabilities = [1]
            return QuantumError(zip(circuits, probabilities))
    else:
        # If T_2 < T_1 we can express this channel as a probabilistic
        # mixture of reset operations and unitary errors:
        circuits = [
            [(IGate(), [0])],
            [(ZGate(), [0])],
            [(Reset(), [0])],
            [(Reset(), [0]), (XGate(), [0])],
        ]
        # Probability
        p_reset0 = p_reset * p0
        p_reset1 = p_reset * p1
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        p_identity = 1 - scale*(p_z + p_reset0 + p_reset1)
        probabilities = [p_identity, scale*p_z, scale*p_reset0, scale*p_reset1]
        return QuantumError(zip(circuits, probabilities))


def new_rescale_thermal_relaxation_error(t1, t2, time, excited_state_population=0,scale=1):
    r"""
    Return a single-qubit thermal relaxation quantum error channel.

    Args:
        t1 (double): the :math:`T_1` relaxation time constant.
        t2 (double): the :math:`T_2` relaxation time constant.
        time (double): the gate time for relaxation error.
        excited_state_population (double): the population of :math:`|1\rangle`
                                           state at equilibrium (default: 0).

    Returns:
        QuantumError: a quantum error object for a noise model.

    Raises:
        NoiseError: If noise parameters are invalid.

    Additional information:
        * For parameters to be valid :math:`T_1` and :math:`T_2` must
          satisfy :math:`T_2 \le 2 T_1`.

        * If :math:`T_2 \le T_1` the error can be expressed as a mixed
          reset and unitary error channel.

        * If :math:`T_1 < T_2 \le 2 T_1` the error must be expressed as a
          general non-unitary Kraus error channel.
    """
    if scale == 0:
        t1 = np.inf
        t2 = np.inf
    else:
        t1 = t1/scale
        t2 = t2/scale

    if excited_state_population < 0:
        raise NoiseError(
            "Invalid excited state population " "({} < 0).".format(excited_state_population)
        )
    if excited_state_population > 1:
        raise NoiseError(
            "Invalid excited state population " "({} > 1).".format(excited_state_population)
        )
    if time < 0:
        raise NoiseError("Invalid gate_time ({} < 0)".format(time))
    if t1 <= 0:
        raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
    if t2 <= 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
    if t2 - 2 * t1 > 0:
        raise NoiseError("Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")

    # T1 relaxation rate
    if t1 == np.inf:
        rate1 = 0
        p_reset = 0
    else:
        rate1 = 1 / t1
        p_reset = 1 - np.exp(-time * rate1)
    # T2 dephasing rate
    if t2 == np.inf:
        rate2 = 0
        exp_t2 = 1
    else:
        rate2 = 1 / t2
        exp_t2 = np.exp(-time * rate2)
    # Qubit state equilibrium probabilities
    p0 = 1 - excited_state_population
    p1 = excited_state_population

    if t2 > t1:
        # If T_2 > T_1 we must express this as a Kraus channel
        # We start with the Choi-matrix representation:
        if scale >0:
            chan = Choi(
                np.array(
                    [
                        [1 - p1 * p_reset, 0, 0, exp_t2],
                        [0, p1 * p_reset, 0, 0],
                        [0, 0, p0 * p_reset, 0],
                        [exp_t2, 0, 0, 1 - p0 * p_reset],
                    ]
                )
            )
            return QuantumError(Kraus(chan))
        elif scale == 0:
            circuits = [[(IGate(), [0])]]
            probabilities = [1]
            return QuantumError(zip(circuits, probabilities))
    else:
        # If T_2 < T_1 we can express this channel as a probabilistic
        # mixture of reset operations and unitary errors:
        circuits = [
            [(IGate(), [0])],
            [(ZGate(), [0])],
            [(Reset(), [0])],
            [(Reset(), [0]), (XGate(), [0])],
        ]
        # Probability
        p_reset0 = p_reset * p0
        p_reset1 = p_reset * p1
        p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        p_identity = 1 - (p_z + p_reset0 + p_reset1)
        probabilities = [p_identity, p_z, p_reset0, p_reset1]
        return QuantumError(zip(circuits, probabilities))

# -----------------------------------------------------------------------------
# Functions to generate basi errors rescaled from device properties

def rescale_basic_device_readout_errors(properties=None, target=None,scale=1):
    """
    Return readout error parameters from either of device Target or BackendProperties.

    If ``target`` is supplied, ``properties`` will be ignored.

    Args:
        properties (BackendProperties): device backend properties
        target (Target): device backend target

    Returns:
        list: A list of pairs ``(qubits, ReadoutError)`` for qubits with
        non-zero readout error values.

    Raises:
        NoiseError: if neither properties nor target is supplied.
    """
    errors = []
    if target is None:
        if properties is None:
            raise NoiseError("Either properties or target must be supplied.")
        # create from BackendProperties
        for qubit, value in enumerate(readout_error_values(properties)):
            if value is not None and not allclose(value, [0, 0]):
                probabilities = [[1 - scale*value[0], scale*value[0]], [scale*value[1], 1 - scale*value[1]]]
                errors.append(([qubit], ReadoutError(probabilities)))
    else:
        # create from Target
        for q in range(target.num_qubits):
            meas_props = target.get("measure", None)
            if meas_props is None:
                continue
            prop = meas_props.get((q,), None)
            if prop is None:
                continue
            if hasattr(prop, "prob_meas1_prep0") and hasattr(prop, "prob_meas0_prep1"):
                p0m1, p1m0 = prop.prob_meas1_prep0, prop.prob_meas0_prep1
            else:
                p0m1, p1m0 = prop.error, prop.error
            probabilities = [[1 - scale*p0m1, scale*p0m1], [scale*p1m0, 1 - scale*p1m0]]
            errors.append(([q], ReadoutError(probabilities)))

    return errors

def rescale_basic_device_gate_errors(
    properties=None,
    gate_error=True,
    thermal_relaxation=True,
    gate_lengths=None,
    gate_length_units="ns",
    temperature=0,
    warnings=None,
    target=None,
    scale=1
):
    
    if properties is None and target is None:
        raise NoiseError("Either properties or target must be supplied.")

    if warnings is not None:
        warn(
            '"warnings" argument has been deprecated as of qiskit-aer 0.12.0 '
            "and will be removed no earlier than 3 months from that release date. "
            "Use the warnings filter in Python standard library instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        warnings = True

    if target is not None:
        if not warnings:
            warn(
                "When `target` is supplied, `warnings` are ignored,"
                " and they are always set to true.",
                UserWarning,
            )

        if gate_lengths:
            raise NoiseError(
                "When `target` is supplied, `gate_lengths` option is not allowed."
                "Use `duration` property in target's InstructionProperties instead."
            )

        return rescale_basic_device_target_gate_errors(
            target=target,
            gate_error=gate_error,
            thermal_relaxation=thermal_relaxation,
            temperature=temperature,
            scale=scale
        )

    # Generate custom gate time dict
    # Units used in the following computation: ns (time), Hz (frequency), mK (temperature).
    custom_times = {}
    relax_params = []
    if thermal_relaxation:
        # If including thermal relaxation errors load
        # T1 [ns], T2 [ns], and frequency [GHz] values from properties
        relax_params = thermal_relaxation_values(properties)
        # Unit conversion: GHz -> Hz
        relax_params = [(t1, t2, freq * 1e9) for t1, t2, freq in relax_params]
        # If we are specifying custom gate times include
        # them in the custom times dict
        if gate_lengths:
            for name, qubits, value in gate_lengths:
                # Convert all gate lengths to nanosecond units
                time = value * _NANOSECOND_UNITS[gate_length_units]
                if name in custom_times:
                    custom_times[name].append((qubits, time))
                else:
                    custom_times[name] = [(qubits, time)]
    # Get the device gate parameters from properties
    device_gate_params = gate_param_values(properties)

    print(device_gate_params)
    # Construct quantum errors
    errors = []
    for name, qubits, gate_length, error_param in device_gate_params:
        # Initilize empty errors
        depol_error = None
        relax_error = None
        # Check for custom gate time
        relax_time = gate_length
        # Override with custom value
        if name in custom_times:
            filtered = [val for q, val in custom_times[name] if q is None or q == qubits]
            if filtered:
                # get first value
                relax_time = filtered[0]
        # Get relaxation error
        if thermal_relaxation:
            relax_error = rescale_device_thermal_relaxation_error(
                qubits, relax_time, relax_params, temperature, scale
            )

        # Get depolarizing error channel
        if gate_error:
            depol_error = rescale_device_depolarizing_error(qubits, error_param, relax_error,scale)

        # Combine errors
        combined_error = _combine_depol_and_relax_error(depol_error, relax_error)
        if combined_error:
            errors.append((name, qubits, combined_error))

    return errors

# -----------------------------------------------------------------------------

# Final function to generate the rescaled noise model from backend


def noise_model_rescaled_backend(
        backend,
        gate_error=True,
        readout_error=True,
        thermal_relaxation=True,
        temperature=0,
        gate_lengths=None,
        gate_length_units="ns",
        scale=1,
        delay_error=True
    ):

    """
    Return a noise model for a device backend, rescaled by a factor scale.
    """

    properties = backend.properties()
    configuration = backend.configuration()
    basis_gates = configuration.basis_gates
    all_qubit_properties = [
        QubitProperties(
            t1=properties.t1(q), t2=properties.t2(q), frequency=properties.frequency(q)
        )
        for q in range(configuration.num_qubits)
    ]
    dt = getattr(configuration, "dt", 0)
    if not properties:
        raise NoiseError(f"Qiskit backend {backend} does not have a BackendProperties")
    

    noise_model = NoiseModel(basis_gates=basis_gates)

    
    # Add single-qubit readout errors
    if readout_error:
        for qubits, error in rescale_basic_device_readout_errors(properties,scale=scale):
            noise_model.add_readout_error(error, qubits)


    gate_errors = rescale_basic_device_gate_errors(
        properties,
        gate_error=gate_error,
        thermal_relaxation=thermal_relaxation,
        gate_lengths=gate_lengths,
        gate_length_units=gate_length_units,
        temperature=temperature,
        scale=scale
        )
    for name, qubits, error in gate_errors:
        noise_model.add_quantum_error(error, name, qubits)


    if delay_error:
        if thermal_relaxation and all_qubit_properties:
            # Add delay errors via RelaxationNiose pass
            try:
                excited_state_populations = [
                    _excited_population(freq=q.frequency, temperature=temperature)
                    for q in all_qubit_properties
                ]
            except BackendPropertyError:
                excited_state_populations = None
            try:
                t1s = [prop.t1 for prop in all_qubit_properties]
                t2s = [_truncate_t2_value(prop.t1, prop.t2) for prop in all_qubit_properties]
                delay_pass = RelaxationNoisePass(
                    t1s=[numpy.inf if x is None else x for x in t1s],  # replace None with np.inf
                    t2s=[numpy.inf if x is None else x for x in t2s],  # replace None with np.inf
                    dt=dt,
                    op_types=Delay,
                    excited_state_populations=excited_state_populations,
                )
                noise_model._custom_noise_passes.append(delay_pass)
            except BackendPropertyError:
                # Device does not have the required T1 or T2 information
                # in its properties
                pass

    return noise_model