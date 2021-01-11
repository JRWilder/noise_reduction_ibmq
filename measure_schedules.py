from qiskit import IBMQ
from qiskit.pulse import Schedule, SetFrequency, ShiftPhase, DriveChannel, Acquire, AcquireChannel, MeasureChannel, Waveform, Gaussian, Play, MemorySlot
from qiskit.pulse.macros import measure, measure_all
from qiskit.test.mock import FakeAlmaden
from qiskit.scheduler import msr

backend = FakeAlmaden()

# Appending a measurement schedule to a Schedule, sched
sched += msr([0, 1], backend) << sched.duration

sched = Schedule(name="Measurement scheduling example")
sched += measure_all(backend)

# For clarity, plot a subset of the channels
sched.draw(plot_range=[0, 1000], channels=[chan(i) for chan in [DriveChannel, MeasureChannel, AcquireChannel] for i in range(3)])

sched = measure(qubits=[0], backend=backend, qubit_mem_slots={0: 1}) # similar to circuit.measure(qubit_reg[0], classical_reg[1]).

# Duration (in number of cycles) for readout
duration = 1600

# Stimulus pulses for qubits 0 and 1
measure_tx = Play(GaussianSquare(duration=duration, amp=0.2, sigma=10, width=duration - 50), MeasureChannel(0))
measure_tx += Play(GaussianSquare(duration=duration, amp=0.2, sigma=10, width=duration - 50), MeasureChannel(1))

measure_rx = Acquire(duration, AcquireChannel(0), MemorySlot(0))
measure_rx += Acquire(duration, AcquireChannel(1), MemorySlot(1))

measure_sched = measure_tx + measure_rx
