from scipy import signal
import wave
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import struct

def bit24_2_32(sub_bytes):
    if sub_bytes[2] < 128:
        return sub_bytes + b'\x00'
    else:
        return sub_bytes + b'\xff'


def read_wav_file(file_path):
    wave_file = wave.open(file_path, 'rb')
    params = wave_file.getparams()
    channels, sampleWidth, frameRate, frames = params[:4]
    data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
    wave_file.close()

    wave_data = np.zeros(channels * frames)
    if sampleWidth == 2:
        wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
    elif sampleWidth == 3:
        samples = np.zeros(channels * frames)
        for i in np.arange(samples.size):
            sub_bytes = data_bytes[i * 3:(i * 3 + 3)]
            sub_bytes = bit24_2_32(sub_bytes)
            samples[i] = struct.unpack('i', sub_bytes)[0]
        wave_data = samples

    wave_data = wave_data.astype(np.float)
    wave_data = np.reshape(wave_data, [frames, channels])

    return wave_data, frameRate


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


def find_click_fdr_tkeo(xn, fs, fl, fwhm, fdr_threshold, ns, snv_threshold = 10):
    if xn.ndim > 1:
        xn = xn[:, 0]

    Wn = fl / fs  # Convert 3 - dB frequency
    b, a = signal.butter(5, Wn, 'highpass')
    xn = signal.filtfilt(b, a, xn)

    back_ground_energy = np.sum(xn ** 2) / len(xn)

    '''
    plt.figure()
    plt.plot(np.arange(0, len(xn)) * (1.0 / fs), xn)
    plt.show()
    '''

    length = len(xn)
    xn_0 = xn[1:(length - 1)]
    xn_1 = xn[0:(length - 2)]
    xnp1 = xn[2:length]

    tkeo = xn_0 * xn_0 - xnp1 * xn_1

    '''
    plt.figure()
    plt.plot(np.arange(0, len(tkeo)) * (1.0 / fs), tkeo)
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
    plt.plot(np.arange(0, len(hMAF1)) * (1.0 / fs), hMAF1)
    plt.plot(np.arange(0, len(hMAF1)) * (1.0 / fs), hMAF2)
    plt.show()
    '''

    hMAF1 = signal.convolve(tkeo, hMAF1)
    hMAF2 = signal.convolve(tkeo, hMAF2)

    length = len(hMAF1)
    hMAF1 = hMAF1[N:(length - N)]
    hMAF2 = hMAF2[N:(length - N)]

    fdr = (hMAF1 - hMAF2) / hMAF1

    for j in range(len(fdr)):
        if fdr[j] < -0.3:
            fdr[j] = -0.3

    '''
    plt.figure()
    plt.plot(np.arange(0, len(hMAF1)), hMAF1)
    plt.plot(np.arange(0, len(hMAF1)), hMAF2)
    plt.show()
    '''

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

            while beg_idx > 0:
                if fdr[beg_idx-1] > 0.3:
                    beg_idx -= 1
                else:
                    break

            while end_idx < len(fdr):
                if fdr[end_idx+1] > 0.3:
                    end_idx += 1
                else:
                    break

            sub_xn = xn[beg_idx:end_idx]
            energy = np.sum(sub_xn ** 2) / len(sub_xn)

            snv = 10 * np.log10(energy/back_ground_energy)

            if snv < snv_threshold:
                beg_idx = -1
                end_idx = -1
                continue

            beg_idx = int(max_idx - ns/2)
            end_idx = int(max_idx + ns/2)
            if beg_idx >= 0 and end_idx <= len(fdr):
                click_index.append([beg_idx, end_idx])

            beg_idx = -1
            end_idx = -1

    return np.array(click_index), xn


if __name__ == '__main__':
    wave_data, frameRate = read_wav_file("Click01.wav")

    fl = 5000
    fwhm = 0.0008
    fdr_threshold = 0.6
    click_index, xn = find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, 256, 8)

    scale = 2 ** 15 / max(xn)
    for i in np.arange(xn.size):
        xn[i] = xn[i] * scale

    for j in range(click_index.shape[0]):
        index = click_index[j]
        # click_data = wave_data[index[0]:index[1], 0]
        click_data = xn[index[0]:index[1]]

        click_data = click_data.astype(np.short)
        filename = "click_%(n)04d.wav" % {'n': j}
        f = wave.open(filename, "wb")
        # set wav params
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(frameRate)
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

