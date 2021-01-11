# ECE 592 Project M3
# Qiskit Pulse x-axis 180 degree rotation
# Name: Johnathan Wilder
# Unity ID: jrwilde2

from qiskit import IBMQ, execute
from qiskit.pulse import Schedule, SetFrequency, ShiftPhase, DriveChannel, Acquire, AcquireChannel, MeasureChannel, Waveform, Gaussian, GaussianSquare, Play, MemorySlot
from qiskit.test.mock import FakeAlmaden
from qiskit.pulse.macros import measure, measure_all
from qiskit.visualization import SchedStyle
from qiskit.scheduler import measure as msr
import matplotlib.pyplot as plt
import numpy as np

# Load IBM account, choose and configure backend
account = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"
dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")
backend_defaults = backend.defaults()


# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

# Create pulse schedule and channels
schedule = Schedule(name='Schedule')
channel0 = DriveChannel(0)

# Setting frequency of channel 0
set_freq = SetFrequency(4.5*GHz, channel0)

# Setting Phase shift of channel 0
phase_pi = ShiftPhase(np.pi, channel0)


print(backend.configuration().meas_map)

# Duration (in number of cycles) for readout
duration = 1600

# Stimulus pulses for qubits 0 and 1
measure_tx = Play(GaussianSquare(duration=duration, amp=0.2, sigma=10, width=duration - 50), MeasureChannel(0))
measure_rx = Acquire(duration, AcquireChannel(0), MemorySlot(0))
measure_sched = measure_tx + measure_rx
schedule |= measure_sched


schedule.draw(plot_range=[0, 1000], channels=[chan(i) for chan in [DriveChannel, MeasureChannel, AcquireChannel] for i in range(3)])
plt.show()
