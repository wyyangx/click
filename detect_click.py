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


if __name__ == '__main__':

    dict = {'0': '', '1': '', '2': ''}
    # dict["0"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    # dict["1"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    # dict["2"] = "/media/ywy/本地磁盘/Data/MobySound/3rd_Workshop/Training_Data/Rissos_(Grampus_grisieus)"

    dict["0"] = "D:/Temp/Training_Data/Blainvilles_beaked_whale_(Mesoplodon_densirostris)"
    dict["1"] = "D:/Temp/Training_Data/Pilot_whale_(Globicephala_macrorhynchus)"
    dict["2"] = "D:/Temp/Training_Data/Rissos_(Grampus_grisieus)"

    for key in dict:
        print(dict[key])
        count = 0
        wav_files = find_click.list_wav_files(dict[key])

        dst_path = "./Data/Click/%(class)s" % {'class': key}
        mkdir(dst_path)
        for pathname in wav_files:

            wave_data, frameRate = find_click.read_wav_file(pathname)

            fl = 5000
            fwhm = 0.0008
            fdr_threshold = 0.62
            click_index, xn = find_click.find_click_fdr_tkeo(wave_data, frameRate, fl, fwhm, fdr_threshold, 256, 8)

            scale = 2 ** 15 / max(xn)
            for i in np.arange(xn.size):
                xn[i] = xn[i] * scale

            for j in range(click_index.shape[0]):
                index = click_index[j]
                # click_data = wave_data[index[0]:index[1], 0]
                click_data = xn[index[0]:index[1]]

                click_data = click_data.astype(np.short)
                filename = "%(path)s/click_%(n)05d.wav" % {'path': dst_path, 'n': count}
                f = wave.open(filename, "wb")
                # set wav params
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(frameRate)
                # turn the data to string
                f.writeframes(click_data.tostring())
                f.close()
                count = count + 1
