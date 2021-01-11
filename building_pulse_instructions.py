from qiskit import IBMQ, execute
from qiskit.pulse import Schedule, Waveform, Play, DriveChannel, Gaussian, SetFrequency, ShiftPhase, Acquire, AcquireChannel, MemorySlot
from qiskit.test.mock import FakeAlmaden
from qiskit.pulse.macros import measure_all
from qiskit.visualization import SchedStyle
import matplotlib.pyplot as plt
import numpy as np

schedule = Schedule(name='Schedule')
channel0 = DriveChannel(0)

# Setting frequency of channel 0
set_freq = SetFrequency(4.5e9, channel0)

# Setting Phase shift of channel 0
phase_pi = ShiftPhase(np.pi, channel0)

# Gaussian Pulse using Parametric Gaussian pulse library
# Using Parametric pulses reduces job size, use when possible
amp = 1
sigma = 10
num_samples = 128
pulse0 = Gaussian(num_samples, amp, sigma, name="Parametric Gaus")

# Adding small delay
delay_5dt = Delay(5, channel0)

# Gaussian Pulse using Waveform and array of samples
times = np.arange(num_samples)
gaussian_samples = np.exp(-1/2 *((times - num_samples / 2) ** 2 / sigma**2))
pulse1 = Waveform(gaussian_samples, name="WF Gaus")

# Using library gaussion function
pulse2 = library.gaussian(duration=num_samples, amp=amp, sigma=sigma, name="Lib Gaus")

# Aquire triggers data aquisistion for read
acquire = Acquire(1200, AcquireChannel(0), MemorySlot(0))
