import cv2
import neurokit2 as nk
from matplotlib import pyplot as plt
import random

for num in range(1):
    bpm = random.randrange(70,90)
    print(bpm)
    simulated_ecg = nk.ecg_simulate(duration=30, sampling_rate=200, heart_rate=bpm)
    print(type(simulated_ecg))
    # print(simulated_ecg)
    print(simulated_ecg.shape)

x= []
for i in range(1,6001):
    x.append(i/200)

ecg_top = simulated_ecg

plt.figure(figsize=(28,28))
plt.axhline(y= 0, color = 'red')
plt.xlabel('');plt.ylabel('')
plt.plot(x, simulated_ecg)

# nk.signal_plot(simulated_ecg, sampling_rate=200)  # Visualize the signal
plt.show()
