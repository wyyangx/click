from scipy import signal
import wave
import numpy as np
import os
import matplotlib.pyplot as plt
import math


def list_wav_files(root_path):
    list = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            (shotname, extension) = os.path.splitext(filename)
            if extension == '.wav':
                list = list + [pathname]

        else:
            list = list + list_wav_files(pathname)

    return list


def find_click_fdr_tkeo(xn, fs, fl, fwhm, fdr_threshold, ns):
    if xn.ndim > 1:
        xn = xn[:, 0]

    Wn = 2 * fl / fs  # Convert 3 - dB frequency
    b, a = signal.butter(8, Wn, 'high')
    xn = signal.filtfilt(b, a, xn)

    length = len(xn)
    xn_0 = xn[1:(length - 1)]
    xn_1 = xn[0:(length - 2)]
    xnp1 = xn[2:length]

    tken = xn_0 * xn_0 - xnp1 * xn_1
    '''
    plt.figure()
    plt.plot(np.arange(0, len(tken)) * (1.0 / frameRate), tken)
    plt.show()
    '''

    # rTK = fwhm / (4 * np.sqrt(np.log(2)))
    rG = fwhm / (2 * np.sqrt(2 * np.log(2)))
    N = np.ceil(5 * rG * fs)
    N = N.astype(np.int32)

    hMAF1 = np.zeros(2*N+1)
    ts = 1 / fs
    for j in range(len(hMAF1)):
        hMAF1[j] = np.exp(-(j - N - 1) * (j - N - 1) * ts * ts / 2 / rG / rG) * ts / rG / np.sqrt(2 * np.pi)

    hMAF2 = np.ones(len(hMAF1)) * sum(hMAF1) / len(hMAF1)

    '''
    plt.figure()
    plt.plot(np.arange(0, len(hMAF1)) * (1.0 / frameRate), hMAF1)
    plt.plot(np.arange(0, len(hMAF1)) * (1.0 / frameRate), hMAF2)
    plt.show()
    '''

    hMAF1 = signal.convolve(tken, hMAF1)
    hMAF2 = signal.convolve(tken, hMAF2)

    length = len(hMAF1)
    hMAF1 = hMAF1[N:(length - N)]
    hMAF2 = hMAF2[N:(length - N)]

    fdr = (hMAF1 - hMAF2) / hMAF1

    for j in range(len(fdr)):
        if fdr[j] < -0.3:
            fdr[j] = -0.3

    '''
    plt.figure()
    plt.plot(np.arange(0, len(fdr)), fdr)
    plt.show()
    '''

    fdr_label = fdr > fdr_threshold

    click_index = []

    beg_idx = -1
    end_idx = -1
    for j in range(len(fdr)):
        if beg_idx == -1 and fdr_label[j]:
            beg_idx = j

        if beg_idx != -1 and not(fdr_label[j]):
            end_idx = j

        if beg_idx != -1 and end_idx != -1:
            max_fdr = 0
            max_idx = beg_idx
            for k in range(beg_idx, end_idx):
                if fdr[k] > max_fdr:
                    max_fdr = fdr[max_idx]
                    max_idx = k

            beg_idx = int(max_idx - ns/2)
            end_idx = int(max_idx + ns/2)
            if beg_idx >= 0 and end_idx <= len(fdr):
                click_index.append([beg_idx, end_idx])

            beg_idx = -1
            end_idx = -1

    return np.array(click_index)


if __name__ == '__main__':
    wave_file = wave.open("Click01.wav", 'rb')
    params = wave_file.getparams()
    channels, sampleWidth, frameRate, frames = params[:4]
    str_data = wave_file.readframes(frames)                    # 读取音频，字符串格式
    wave_file.close()
    wave_data = np.fromstring(str_data, dtype=np.int16)         # 将字符串转化为int
    wave_data = np.reshape(wave_data, [frames, channels])

    fl = 5000
    fwhm = 0.0004
    fdr_threshold = 0.6
    click_index = find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, 128)

    for j in range(click_index.shape[0]):
        index = click_index[j]
        click_data = wave_data[index[0]:index[1], 0]

        click_data = click_data.astype(np.short)
        filename = "click_%(n)04d.wav" % {'n': j}
        f = wave.open(filename, "wb")
        # set wav params
        f.setnchannels(1)
        f.setsampwidth(sampleWidth)
        f.setframerate(frameRate)
        # turn the data to string
        f.writeframes(click_data.tostring())
        f.close()

        '''
        plt.figure(j)
        plt.plot(np.arange(0, len(click_data)) * (1.0 / frameRate), click_data)
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.grid('True')  # 标尺，on：有，off:无。
        plt.show()
        '''

