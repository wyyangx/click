from scipy import signal
import wave
import numpy as np
import matplotlib.pyplot as plt
import math

wave_file = wave.open("./Data/Click/2/click_00010.wav", 'rb')
params = wave_file.getparams()
channels, sampleWidth, frameRate, frames = params[:4]
data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
wave_file.close()
wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
wave_data = np.reshape(wave_data, [frames, channels])

plt.figure()
plt.plot(np.arange(0, len(wave_data)) * (1.0 / frameRate), wave_data)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid('True')  # 标尺，on：有，off:无。
plt.show()