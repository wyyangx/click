import os
import wave
import numpy as np
import find_click
import struct


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 / 符号
    path = path.rstrip("/")

    # 判断路径是否存在
    if not os.path.exists(path):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)


def bit24_2_32(sub_bytes):
    if sub_bytes[2] < 128:
        return sub_bytes + b'\x00'
    else:
        return sub_bytes + b'\xff'


if __name__ == '__main__':

    dict = {'0': '', '1': '', '2': ''}
    dict["0"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Rissos_(Grampus_grisieus)"

    for key in dict:
        print(dict[key])
        count = 0
        wav_files = find_click.list_wav_files(dict[key])

        dst_path = "./Data/Click/%(class)s" % {'class': key}
        mkdir(dst_path)
        for pathname in wav_files:
            wave_file = wave.open(pathname, 'rb')
            params = wave_file.getparams()
            channels, sampleWidth, frameRate, frames = params[:4]
            data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
            wave_file.close()

            int2float = 1
            wave_data = np.zeros(channels*frames)
            if sampleWidth == 2:
                int2float = (2 ** 15) - 1
                wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
            elif sampleWidth == 3:
                int2float = (2**23) - 1
                '''
                new_data_bytes = b''
                for i in np.arange(channels*frames):
                    sub_bytes = data_bytes[i*3:(i*3+3)]
                    sub_bytes = bit24_2_32(sub_bytes)
                    new_data_bytes += sub_bytes
                wave_data = np.fromstring(new_data_bytes, dtype=np.int32)  # 将字符串转化为int
                '''
                samples = np.zeros(channels*frames)
                for i in np.arange(samples.size):
                    sub_bytes = data_bytes[i*3:(i*3+3)]
                    sub_bytes = bit24_2_32(sub_bytes)
                    samples[i] = struct.unpack('i', sub_bytes)[0]
                wave_data = samples

            '''                
            for i in np.arange(wave_data.size):
                wave_data[i] = wave_data[i]/int2float
            '''

            wave_data = np.reshape(wave_data, [frames, channels])

            fl = 5000
            fwhm = 0.0004
            fdr_threshold = 0.62
            click_index = find_click.find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, 128)

            for j in range(click_index.shape[0]):
                index = click_index[j]
                click_data = wave_data[index[0]:index[1], 0]

                click_data = click_data.astype(np.short)
                filename = "%(path)s/click_%(n)04d.wav" % {'path': dst_path, 'n': count}
                f = wave.open(filename, "wb")
                # set wav params
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(frameRate)
                # turn the data to string
                f.writeframes(click_data.tostring())
                f.close()
                count = count + 1
