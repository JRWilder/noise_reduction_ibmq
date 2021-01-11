from qiskit.pulse import Schedule, Play, Constant, Delay, DriveChannel
import matplotlib.pyplot as plt

sched_a = Schedule(name="A")
sched_b = Schedule(name="B")

sched_a = sched_a.insert(0, Play(Constant(duration=5, amp=1), DriveChannel(0)))
sched_a |= Play(Constant(duration=5, amp=0.5), DriveChannel(0)).shift(10)
sched_a.draw()

sched_b |= Play(Constant(duration=5, amp=-1), DriveChannel(0))
sched_b |= Play(Constant(duration=5, amp=-0.5), DriveChannel(0)) << 10
sched_b.draw()

sched_a_and_b = sched_a.insert(20, sched_b)  # A followed by B at time 20
sched_b_and_a = sched_a | sched_b << 20      # B followed by A at time 20
sched_b_and_a.draw()

sched_a_plus_b = sched_a.append(sched_b)
sched_a_plus_b.draw()

plt.show()
