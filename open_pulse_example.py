from qiskit import IBMQ, execute
from qiskit.pulse import Schedule, Waveform, Play, DriveChannel
from qiskit.test.mock import FakeAlmaden
from qiskit.pulse.macros import measure_all
from qiskit.visualization import SchedStyle
import matplotlib.pyplot as plt

sched = Schedule(name='Schedule')
my_pulse = Waveform([0.00043, 0.0007 , 0.00112, 0.00175, 0.00272, 0.00414, 0.00622,
                        0.00919, 0.01337, 0.01916, 0.02702, 0.03751, 0.05127, 0.06899,
                        0.09139, 0.1192 , 0.15306, 0.19348, 0.24079, 0.29502, 0.35587,
                        0.4226 , 0.49407, 0.56867, 0.64439, 0.71887, 0.78952, 0.85368,
                        0.90873, 0.95234, 0.98258, 0.99805, 0.99805, 0.98258, 0.95234,
                        0.90873, 0.85368, 0.78952, 0.71887, 0.64439, 0.56867, 0.49407,
                        0.4226 , 0.35587, 0.29502, 0.24079, 0.19348, 0.15306, 0.1192 ,
                        0.09139, 0.06899, 0.05127, 0.03751, 0.02702, 0.01916, 0.01337,
                        0.00919, 0.00622, 0.00414, 0.00272, 0.00175, 0.00112, 0.0007 ,
                        0.00043],
                       name="short_gaussian_pulse")
qubit_idx = 0
sched = sched.insert(0, Play(my_pulse, DriveChannel(qubit_idx)))

backend = IBMQ.load_account().get_backend(open_pulse=True)
sched = sched.insert(sched.duration, measure_all(backend))
style = SchedStyle(figsize=(3, 2), title_font_size=10, axis_font_size=8)
sched.draw(plot_range=[0, 500], style=style)
plt.show()

#plt.show(sched.draw(plot_range=[0, 1000], style=SchedStyle(axis_font_size=8)))

job = execute(sched, backend)
# result = job.result()
# print(result)
