import os
import wave
import numpy as np
import find_click


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


def resample(input_signal, src_fs, tar_fs):
    """
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    """

    if src_fs == tar_fs:
        return input_signal

    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0 * (audio_len - 1) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len, endpoint=False) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs+0.5), np.int(audio_time_max * tar_fs+0.5), endpoint=False) / tar_fs
    output_signal = np.interp(tar_time, src_time, input_signal).astype(dtype)

    return output_signal


def cut_data(input_signal, out_len):

    audio_len = len(input_signal)
    if audio_len < out_len:
        return input_signal

    beg_idx = int(audio_len/2 - out_len / 2)
    end_idx = beg_idx + out_len

    return input_signal[beg_idx:end_idx]


if __name__ == '__main__':

    tar_fs = 192000
    signal_len = 320

    '''
    dict = {'0': '', '1': '', '2': ''}
    dict["0"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/" \
                "Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Rissos_(Grampus_grisieus)"    

    # dict["0"] = "D:/Temp/Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    # dict["1"] = "D:/Temp/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    # dict["2"] = "D:/Temp/Training_Data/Rissos_(Grampus_grisieus)"
    '''

    dict = {'0': '', '1': '', '2': '', '3':'', '4':'', '5':'', '6':'', '7':''}

    dict["3"] = "/media/ywy/本地磁盘/Data/MobySound/5th_Workshop/5th_DCL_data_bottlenose"
    dict["4"] = "/media/ywy/本地磁盘/Data/MobySound/5th_Workshop/5th_DCL_data_common/DC"
    dict["5"] = "/media/ywy/本地磁盘/Data/MobySound/5th_Workshop/5th_DCL_data_common/DD"
    dict["6"] = "/media/ywy/本地磁盘/Data/MobySound/5th_Workshop/5th_DCL_data_melon-headed"
    dict["7"] = "/media/ywy/本地磁盘/Data/MobySound/5th_Workshop/5th_DCL_data_spinner"

    dict["0"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/" \
                "Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Rissos_(Grampus_grisieus)"


    for key in dict:
        print(dict[key])
        count = 0
        wav_files = find_click.list_wav_files(dict[key])

        dst_path = "./Data/ClickC8/%(class)s" % {'class': key}
        mkdir(dst_path)

        for pathname in wav_files:

            print(pathname)
            wave_data, frameRate = find_click.read_wav_file(pathname)

            fl = 5000
            fwhm = 0.0008
            fdr_threshold = 0.62
            click_index, xn = find_click.find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, signal_len, 8)

            scale = (2 ** 15 - 1) / max(xn)
            for i in np.arange(xn.size):
                xn[i] = xn[i] * scale

            for j in range(click_index.shape[0]):
                index = click_index[j]
                # click_data = wave_data[index[0]:index[1], 0]

                click_data = xn[index[0]:index[1]]

                click_data = resample(click_data, frameRate, tar_fs)  #

                click_data = cut_data(click_data, signal_len)

                click_data = click_data.astype(np.short)
                filename = "%(path)s/click_%(n)06d.wav" % {'path': dst_path, 'n': count}
                f = wave.open(filename, "wb")
                # set wav params
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(tar_fs)
                # turn the data to string
                f.writeframes(click_data.tostring())
                f.close()
                count = count + 1

            if count > 20000:
                break

        print("count = %(count)d" % {'count': count})
