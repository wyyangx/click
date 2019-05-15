from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt
import math

n = 50

clicks = np.load("./Data/ClickC8npy_2006/6/palmyra102006-061026-211000_4_N413.npy")
frameRate = 192000
wave_data = clicks[n]

print(clicks.shape)

plt.figure()
plt.plot(np.arange(0, len(wave_data)) * (1.0 / frameRate), wave_data, color='b')
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid('True')  # 标尺，on：有，off:无。
plt.show()
