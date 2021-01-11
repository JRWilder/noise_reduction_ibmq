from pathlib import Path
import time

# Disable warning from qiskit
import warnings
warnings.filterwarnings('ignore')

# Choose to run experiments or to use saved data
use_saved_data = True
use_IBM = False

if use_saved_data == False:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print("Time label for data saved throughout this experiment:" + timestr)
data_folder = Path("resources/superconducting-qubits-improving-the-performance-of-single-qubit-gates/")

# General imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy import interpolate
import jsonpickle


# Parameters
giga = 1.0e9
mega = 1.0e6
micro = 1.0e-6
nano = 1.0e-9
scale_factor = 1e-14
colors = {"Q-CTRL_2": "#BF04DC", "Q-CTRL": "#680CE9", "Square": "#000000", 'IBM default X-gate':'r'}

# Q-CTRL imports
from qctrl import Qctrl
from qctrlvisualizer import plot_controls
from qctrlvisualizer import get_qctrl_style

plt.style.use(get_qctrl_style())
bloch_prop_cycle = plt.rcParams['axes.prop_cycle']
bloch_colors = bloch_prop_cycle.by_key()['color']
bloch_markers = {'x': 'x', 'y': 's', 'z':'o'}
bloch_lines = {'x': '--', 'y': '-.', 'z':'-'}
bloch_basis = ['x', 'y', 'z']

# Q-CTRL auxiliary functions
def get_detuning_scan(scan_array, control):

    durations = [segment["duration"] for segment in control["I"]]
    duration = np.sum(durations)

    I_values = [segment["value"] for segment in control["I"]]
    Q_values = [segment["value"] for segment in control["Q"]]

    # Define system object
    system = qctrl.factories.systems.create(name="qubit", hilbert_space_dimension=2)

    initial_state = np.array([1.0, 0.0])

    shift_I = qctrl.factories.shift_controls.create(
        name="I", system=system, operator=sigma_x / 2.0)
    shift_I_pulse = qctrl.factories.custom_pulses.create(
        control=shift_I,
        segments=[{"duration": d, "value": v} for d, v in zip(durations, I_values)])

    shift_Q = qctrl.factories.shift_controls.create(
        name="Q", system=system, operator=sigma_y / 2.0)
    shift_Q_pulse = qctrl.factories.custom_pulses.create(
        control=shift_Q,
        segments=[{"duration": d, "value": v} for d, v in zip(durations, Q_values)])

    drifts = qctrl.factories.additive_noises.create(
        name="drift", system=system, operator=sigma_z)

    quasi_static_function = qctrl.factories.quasi_static_functions.create(
        x_noise=drifts, x_coefficients=scan_array)

    # Define target as the excited state population
    target = qctrl.factories.targets.create(
        system=system, unitary_operator=np.eye(2), projection_operator=np.array([[1, 0], [0, 0]]))

    # Compute quasi-static function
    result = qctrl.services.quasi_static_functions.calculate(quasi_static_function,)

    # Extract infidelities (excited state population)
    noises_and_infidelities = np.array(
        [[sampled_point["coefficients"][0], sampled_point["infidelity"]]
         for sampled_point in result.sampled_points])

    noises_and_infidelities_sorted = noises_and_infidelities[
        noises_and_infidelities.argsort(axis=0)[:, 0]]

    # Return infidelity (ground state population)
    return 1 - noises_and_infidelities_sorted[:, 1]


def get_amplitude_scan(amplitudes, control):
    durations = [segment["duration"] for segment in control["I"]]
    duration = np.sum(durations)
    I_values = np.array([segment["value"] for segment in control["I"]])
    Q_values = np.array([segment["value"] for segment in control["Q"]])
    phasors = I_values + Q_values* 1j
    # Define system object
    system = qctrl.factories.systems.create(name="qubit", hilbert_space_dimension=2)

    initial_state = np.array([1.0, 0.0])

    drive = qctrl.factories.drive_controls.create(
        name="sigma_m", system=system, operator=sigma_m / 2)

    drive_pulse = qctrl.factories.custom_pulses.create(
        control=drive,
        segments=[
            {"duration": d, "value": v} for d,v in zip(durations,phasors)])

    amplitude_noise = qctrl.factories.control_noises.create(
        name="amplitude", control=drive)

    quasi_static_function = qctrl.factories.quasi_static_functions.create(
        x_noise=amplitude_noise, x_coefficients=amplitudes)

    # Define target. As above, this target allows us to calculate the ground
    # state populations.
    target = qctrl.factories.targets.create(
        system=system, unitary_operator=np.eye(2), projection_operator=np.array([[1, 0], [0, 0]]))

    # Compute quasi-static function
    result = qctrl.services.quasi_static_functions.calculate(quasi_static_function,)

    # Extract infidelities
    noises_and_infidelities = np.array(
        [[sampled_point["coefficients"][0], sampled_point["infidelity"]]
         for sampled_point in result.sampled_points])

    noises_and_infidelities_sorted = noises_and_infidelities[
        noises_and_infidelities.argsort(axis=0)[:, 0]]

    # Return infidelities
    return 1 - noises_and_infidelities_sorted[:, 1]


def simulation_coherent(control, time_samples):

    durations = [segment["duration"] for segment in control["I"]]
    I_values = np.array([segment["value"] for segment in control["I"]])
    Q_values = np.array([segment["value"] for segment in control["Q"]])
    duration = sum(durations)
    initial_state_vector=np.array([1.0, 0.0])
    sample_times=np.linspace(0, duration, time_samples)

    system = qctrl.factories.systems.create(
        name='Single qubit simulation',
        hilbert_space_dimension=2)

    target = qctrl.factories.targets.create(
    system=system,

    unitary_operator=X90_gate,
    projection_operator=np.identity(2))

    phasors = I_values + 1j*Q_values
    drive = qctrl.factories.drive_controls.create(
        name="sigma_m", system=system, operator=sigma_m / 2)

    drive_pulse = qctrl.factories.custom_pulses.create(
        control=drive,
        segments=[
            {"duration": d, "value": v} for d,v in zip(durations,phasors)])

    simulation = qctrl.factories.coherent_simulations.create(
        system=system,
        point_times=np.linspace(0, duration, time_samples),
        initial_state_vector=np.array([1., 0.]))

    # Run simulation
    simulation_result = qctrl.services.coherent_simulations.run(system)

    # Extract results
    gate_times = np.array([frame['time']
                  for frame in simulation_result.simulations[0].trajectories[0].frames])
    state_vectors = np.array([frame['state_vector']
                          for frame in simulation_result.simulations[0].trajectories[0].frames])
    infidelities = np.array([frame['infidelity']
                          for frame in simulation_result.simulations[0].trajectories[0].frames])
    bloch_vector_components = {'x': np.real(np.array([np.linalg.multi_dot(
                                   [np.conj(state), sigma_x, state])
                                                      for state in state_vectors])),
                               'y': np.real(np.array([np.linalg.multi_dot(
                                   [np.conj(state), sigma_y, state])
                                                      for state in state_vectors])),
                               'z': np.real(np.array([np.linalg.multi_dot(
                                   [np.conj(state), sigma_z, state])
                                                      for state in state_vectors]))}

    return infidelities, bloch_vector_components, gate_times


def save_var(file_name, var):
    # Save a single variable to a file using jsonpickle
    f = open(file_name, 'w+')
    to_write = jsonpickle.encode(var)
    f.write(to_write)
    f.close()

def load_var(file_name):
    # Return a variable from a json file
    f = open(file_name, 'r+')
    encoded = f.read()
    decoded = jsonpickle.decode(encoded)
    f.close()
    return decoded


# Q-CTRL login
qctrl = Qctrl(email='johnathan.wilder3698@gmail.com', password='Jdubya@98')

# IBM-Q imports
if use_IBM == True:
    from qiskit import IBMQ

    from qiskit.tools.jupyter import *
    from qiskit.tools.monitor import job_monitor

    from qiskit.compiler import assemble
    from qiskit.pulse.library import Waveform

    import qiskit.pulse as pulse

    from qiskit.pulse import (DriveChannel, MeasureChannel,Play,
                              AcquireChannel, Acquire, MemorySlot)

    # IBM credentials and backend selection
    # IBMQ.enable_account('your IBM token')
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='180 Z rotation')

    backend = provider.get_backend("ibmq_armonk")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    # Backend properties
    qubit = 0
    qubit_freq_est = backend_defaults.qubit_freq_est[qubit]
    dt = backend_config.dt
    print(f"Qubit: {qubit}")
    print(f"Hardware sampling time: {dt/nano} ns")
    print(f"Qubit frequency estimate: {qubit_freq_est/giga} GHz")

    # Drive and measurement channels
    drive_chan = DriveChannel(qubit)
    meas_chan = MeasureChannel(qubit)
    inst_sched_map = backend_defaults.instruction_schedule_map
    measure_schedule = inst_sched_map.get('measure', qubits=backend_config.meas_map[qubit])
else:
    # Use default dt for armonk backend
    dt = 2/9 * 1e-9
    # Use 0 qubit for armonk
    qubit = 0
    # Use last frequency update
    qubit_freq_est = 4974444325.742604
    qubit_frequency_updated = qubit_freq_est
    print("IBM offline parameters")
    print(f"Qubit: {qubit}")
    print(f"Hardware sampling time: {dt/nano} ns")
    print(f"Qubit frequency estimate: {qubit_freq_est/giga} GHz")
