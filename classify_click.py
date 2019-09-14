import tensorflow as tf
import find_click
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import copy
#
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle


class ClickClassifier:
    def __init__(self, crop_len, n_classes, data_name, odontocete, fold_num=3):
        self.crop_len = crop_len  # cnn 网络输入的特征长度
        self.n_classes = n_classes
        self.cnn_ftu_len = int(crop_len / 2)
        #
        self.train_data = []  # 类型list,共 n_classes 个元素,每个元素是一个list（包含两个元素,分别是click数据和label）
        self.test_data = []  # 和 test_data 的数据格式一样

        self.train_click = []
        self.train_ys = []
        #
        self.fold_num = fold_num
        self.data_name = data_name
        self.odontocete = odontocete
        #
        self.test_ys = []  # 类型list,每组click的label
        self.test_click_batch = []  # 类型list,每batch_click_num个click一组进行识别
        #
        self.cnn_mv_scores = np.empty((0, n_classes))  # 每生成测试数据,单次测试的结果,保存每个测试样本属于每个类的分数值
        self.cnn_mp_scores = np.empty((0, n_classes))  # shape为(n, n_classes), n为测试样本数
        self.cnn_labels = np.empty((0, n_classes))
        #
        self.cnn_mv_rates = np.empty((0, n_classes))  # calculate_rates 函数中记录每次测试的识别率,shape为(1, n_classes) 
        self.cnn_mp_rates = np.empty((0, n_classes))
        #
        self.gmm_models = []
        self.gmm_scores = np.empty((0, n_classes))
        self.gmm_labels = np.empty((0, n_classes))
        #
        self.gmm_rates = np.empty((0, n_classes))  # calculate_rates 函数中记录每次测试的识别率,shape为(1, n_classes) 

    # 将每次测试结果保存到文件中,多次的测试结果保存在一个文件中
    def save(self, batch_click_num):
        # cnn
        file_path = "./Data/result/%s_cnn_mv_score_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.cnn_mv_scores)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.cnn_mv_scores))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_cnn_mp_score_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.cnn_mp_scores)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.cnn_mp_scores))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_cnn_label_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.cnn_labels)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.cnn_labels))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_cnn_mv_rates_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.cnn_mv_rates)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.cnn_mv_rates))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_cnn_mp_rates_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.cnn_mp_rates)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.cnn_mp_rates))
            np.save(file_path, tmp)

        # gmm
        file_path = "./Data/result/%s_gmm_score_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.gmm_scores)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.gmm_scores))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_gmm_label_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.gmm_labels)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.gmm_labels))
            np.save(file_path, tmp)

        file_path = "./Data/result/%s_gmm_rates_(b%d).npy" % (self.data_name, batch_click_num)
        if not os.path.exists(file_path):
            np.save(file_path, self.gmm_rates)
        else:
            tmp = np.load(file_path)
            tmp = np.vstack((tmp, self.gmm_rates))
            np.save(file_path, tmp)

    # 加载测试结果文件,文件中保存的测试结果次数可以通过cnn_mv_rates,cnn_mp_scores,或
    # gmm_rates的shape获取
    def load_results(self, batch_click_num):
        # cnn
        file_path = "./Data/result/%s_cnn_mv_score_(b%d).npy" % (self.data_name, batch_click_num)
        self.cnn_mv_scores = np.load(file_path)
        file_path = "./Data/result/%s_cnn_mp_score_(b%d).npy" % (self.data_name, batch_click_num)
        self.cnn_mp_scores = np.load(file_path)
        file_path = "./Data/result/%s_cnn_label_(b%d).npy" % (self.data_name, batch_click_num)
        self.cnn_labels = np.load(file_path)
        file_path = "./Data/result/%s_cnn_mv_rates_(b%d).npy" % (self.data_name, batch_click_num)
        self.cnn_mv_rates = np.load(file_path)
        file_path = "./Data/result/%s_cnn_mp_rates_(b%d).npy" % (self.data_name, batch_click_num)
        self.cnn_mp_rates = np.load(file_path)
        # gmm
        file_path = "./Data/result/%s_gmm_score_(b%d).npy" % (self.data_name, batch_click_num)
        self.gmm_scores = np.load(file_path)
        file_path = "./Data/result/%s_gmm_label_(b%d).npy" % (self.data_name, batch_click_num)
        self.gmm_labels = np.load(file_path)
        file_path = "./Data/result/%s_gmm_rates_(b%d).npy" % (self.data_name, batch_click_num)
        self.gmm_rates = np.load(file_path)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_1x2(x):
        return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def split_data(xs):
        num = xs.shape[0]
        split_idx = int(num * 4 / 5)
        xs0 = xs[0:split_idx, :]
        xs1 = xs[split_idx:num, :]
        return xs0, xs1

    @staticmethod
    def random_crop(xs, batch_click_num, n_total):
        num = xs.shape[0]
        rc_xs = np.empty((0, 192))

        for i in range(0, n_total):
            for j in range(batch_click_num * i, batch_click_num * (i + 1)):
                index = j % num
                temp_x = xs[index]
                # beg_idx = np.random.randint(0, 32)
                beg_idx = np.random.randint(64, (64 + 32))
                crop_x = temp_x[beg_idx:(beg_idx + 192)]
                crop_x = np.reshape(crop_x, [1, 192])
                rc_xs = np.vstack((rc_xs, crop_x))

        return rc_xs

    @staticmethod
    def split_data(xs, ratio=0.5, rd=True):
        if not rd:
            num = len(xs)
            split_idx = int(num * ratio)
            xs0 = xs[0:split_idx]
            xs1 = xs[split_idx:]
            return xs0, xs1

        split_idx = int(len(xs) * ratio)
        ri = np.random.permutation(len(xs))
        ri0 = ri[0:split_idx]
        ri1 = ri[split_idx:]
        xs0 = [xs[i] for i in ri0]
        xs1 = [xs[i] for i in ri1]
        return xs0, xs1

    def load_data(self, data_path):

        self.cnn_mv_scores = []
        self.cnn_mp_scores = []
        self.cnn_labels = []
        #
        self.gmm_models = []
        self.gmm_scores = []
        self.gmm_labels = []

        self.train_data = []
        self.test_data = []

        for c in range(0, self.n_classes):
            print("---------------------------------------------------------")
            path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}

            npy_files = find_click.list_files(path, '.npy')
            print("load data : %s, the number of files : %d" % (path, len(npy_files)))
            print("---------------------------------------------------------")

            random_index = np.random.permutation(len(npy_files))

            count = 0
            clicks_train = np.empty((0, 320))
            clicks_test = np.empty((0, 320))

            for idx in range(len(npy_files)):
                index = random_index[idx]
                npy_file = npy_files[index]

                clicks = np.load(npy_file)
                count += clicks.shape[0]

                if idx < len(npy_files) * (self.fold_num - 1)/self.fold_num:
                    clicks_train = np.vstack((clicks_train, clicks))
                else:
                    clicks_test = np.vstack((clicks_test, clicks))

            label = c
            label = np.array([label])
            label = list(label)

            clicks_train = list(np.expand_dims(clicks_train, axis=0))
            clicks_train = clicks_train + label

            clicks_test = list(np.expand_dims(clicks_test, axis=0))
            clicks_test = clicks_test + label

            print("the number of clicks : %(n)d" % {'n': count})

            self.train_data.append(clicks_train)
            self.test_data.append(clicks_test)

    @staticmethod
    def shufflelists(xs, ys, num):
        shape = xs.shape
        ri = np.random.permutation(shape[0])
        ri = ri[0: num]
        batch_xs = np.empty((0, xs.shape[1]))
        batch_ys = np.empty((0, ys.shape[1]))
        for i in ri:
            batch_xs = np.vstack((batch_xs, xs[i]))
            batch_ys = np.vstack((batch_ys, ys[i]))

        return batch_xs, batch_ys

    def cnn_net(self):
        tf.reset_default_graph()

        x_in = tf.placeholder("float", [None, self.cnn_ftu_len])
        y_label = tf.placeholder("float", [None, self.n_classes])

        # 输入
        x_input = tf.reshape(x_in, [-1, 1, self.cnn_ftu_len, 1])

        '''
        # 第一个卷积层
        W_conv1 = self.weight_variable([1, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_input, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_1x2(h_conv1)

        # 第二个卷积层
        W_conv2 = self.weight_variable([1, 5, 32, 32])
        b_conv2 = self.bias_variable([32])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_1x2(h_conv2)

        # 密集链接层
        W_fc1 = self.weight_variable([1 * int(self.ftu_len / 4) * 32, 256])
        b_fc1 = self.bias_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * int(self.ftu_len / 4) * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

        # 输出层
        W_fc2 = self.weight_variable([256, self.n_classes])
        b_fc2 = self.bias_variable([self.n_classes])
        weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_out = tf.nn.softmax(weight)
        '''

        # 第一个卷积层
        W_conv1 = self.weight_variable([1, 9, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_input, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_1x2(h_conv1)

        # 第二个卷积层
        W_conv2 = self.weight_variable([1, 9, 32, 32])
        b_conv2 = self.bias_variable([32])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_1x2(h_conv2)

        # 第3个卷积层
        W_conv3 = self.weight_variable([1, 5, 32, 32])
        b_conv3 = self.bias_variable([32])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_1x2(h_conv3)

        # 密集链接层1
        W_fc1 = self.weight_variable([1 * int(self.cnn_ftu_len / 8) * 32, 128])
        b_fc1 = self.bias_variable([128])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 1 * int(self.cnn_ftu_len / 8) * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        # 密集链接层2
        W_fc2 = self.weight_variable([128, 64])
        b_fc2 = self.bias_variable([64])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # Dropout
        keep_prob = tf.placeholder("float")
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=keep_prob)

        # 输出层
        W_fc3 = self.weight_variable([64, self.n_classes])
        b_fc3 = self.bias_variable([self.n_classes])
        weight = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
        y_out = tf.nn.softmax(weight)

        #

        cross_entropy = -tf.reduce_sum(y_label * tf.log(y_out))

        # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return x_in, y_label, keep_prob, y_out, train_step, accuracy, weight

    def generate_train_data(self, rand_margin=24, n_total=20000):
        print("Generate train data ... ... ")

        sample_num = n_total * self.n_classes
        self.train_ys = np.zeros((sample_num, self.n_classes))

        crop_len = self.crop_len
        self.train_click = np.zeros((sample_num, crop_len))

        sample_index = 0
        for c in range(0, self.n_classes):
            data = self.train_data[c]
            clicks = data[0].copy() # 这里必须进行深拷贝,否则浅拷贝后面能量归一化会改变self.train_data中的数据
            click_num = clicks.shape[0]  # Click 数量
            label = np.zeros(self.n_classes)
            label[c] = 1

            org_len = clicks.shape[1]
            ref_idx = int((org_len - crop_len) / 2 - rand_margin / 2)

            n_step = int(click_num / n_total)
            if n_step == 0:
                n_step = 1

            rand_index = np.random.permutation(click_num)  # 将Click样本打乱,使得训练样本可以从多个不同文件中的Click产生

            click_index = 0
            for n in range(0, n_total):
                click = clicks[rand_index[click_index]]

                beg_idx = ref_idx + np.random.randint(0, rand_margin)
                crop_x = click[beg_idx:(beg_idx + crop_len)]
                crop_x = np.reshape(crop_x, [1, crop_len])

                energy = np.sqrt(np.sum(crop_x ** 2))
                crop_x /= energy

                self.train_ys[sample_index, :] = label
                self.train_click[sample_index, :] = crop_x

                click_index = click_index + n_step
                click_index = click_index % click_num
                sample_index += 1

    def train_gmm(self, components=16):
        print("\r\nTrain gmm ... ...")
        self.gmm_models = []
        train_click = self.train_click.copy()
        click_feature = self.feature_extractor_gmm(train_click)

        for c in range(0, self.n_classes):
            index = self.train_ys[:, c] == 1
            feature = click_feature[index, :]
            gmm = GaussianMixture(n_components=components, covariance_type='diag').fit(feature)
            self.gmm_models.append(gmm)

    def test_gmm(self):
        click_batchs = copy.deepcopy(self.test_click_batch)
        test_ys = self.test_ys

        print("Test gmm ..., batch_click_num = %d ..." % click_batchs[0].shape[0])
        gmm_score = np.zeros((len(test_ys), self.n_classes))
        ref_y = np.zeros((len(test_ys), self.n_classes))

        for i in range(len(click_batchs)):
            clicks = click_batchs[i]
            features = self.feature_extractor_gmm(clicks)
            #
            batch_click_num = clicks.shape[0]
            score = np.zeros((batch_click_num, self.n_classes))
            for c in range(self.n_classes):
                class_scores = self.gmm_models[c].score_samples(features)
                score[:, c] = class_scores

            score = np.sum(score, axis=0)
            score = self.soft_max(score)
            gmm_score[i, :] = score
            ref_y[i, :] = np.array(test_ys[i])

        self.gmm_scores = gmm_score
        self.gmm_labels = ref_y

    @staticmethod
    def soft_max(score):
        max_v = np.max(score)
        score = score - max_v
        score = np.exp(score) / np.sum(np.exp(score))
        return score

    def feature_extractor_cnn(self, clicks):
        hanming_win = np.hamming(clicks.shape[1])
        hanming_win = np.tile(hanming_win, (clicks.shape[0], 1))
        xs = clicks * hanming_win

        x_fft = np.fft.fft(xs, axis=1)
        x_fft = np.abs(x_fft)
        features = x_fft[:, range(self.cnn_ftu_len)]  # 由于对称性,只取一半区间

        return features

    @staticmethod
    def feature_extractor_gmm(clicks):
        hanming_win = np.hamming(clicks.shape[1])

        hanming_win = np.tile(hanming_win, (clicks.shape[0], 1))
        xs = clicks * hanming_win
        x_fft = np.fft.fft(xs, 2048, axis=1)
        x_fft = np.sqrt(np.abs(x_fft))

        crop_x = x_fft[:, 0:1024]
        crop_x = np.fft.fft(np.log(crop_x), axis=1)
        crop_x = np.sqrt(np.abs(crop_x))
        crop_x = crop_x[:, 1:15]
        features = crop_x

        '''
        features = np.zeros((clicks.shape[0], 14))
        for i in range(0, clicks.shape[0]):
            xs = clicks[i] * hanming_win
            x_fft = np.fft.fft(xs, 2048)
            x_fft = np.sqrt(np.abs(x_fft))

            crop_x = x_fft[:1024]
            crop_x = np.fft.fft(np.log(crop_x))
            crop_x = np.sqrt(np.abs(crop_x))
            crop_x = crop_x[1:15]
            features[i, :] = crop_x
        '''
        return features

    @staticmethod
    def shuffle_data(xs, ys, num=-1):
        shape = xs.shape
        ri = np.random.permutation(shape[0])
        if num <= 0:
            batch_xs = xs[ri, :]
            batch_ys = ys[ri, :]
        else:
            ri = ri[0:num]
            batch_xs = xs[ri, :]
            batch_ys = ys[ri, :]

        return batch_xs, batch_ys

    def train_cnn(self, epochs, batch_size=256):
        print("Train cnn ... ...")
        x_in, y_label, keep_prob, y_out, train_step, accuracy, weight = self.cnn_net()
        saver = tf.train.Saver()

        train_click = self.train_click.copy()
        train_xs = self.feature_extractor_cnn(train_click)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pre_rate = 0
            for e in range(epochs):
                xs, ys = self.shuffle_data(train_xs, self.train_ys)

                for i in range(train_xs.shape[0] // batch_size):
                    beg_i = batch_size * i
                    end_i = batch_size * (i + 1)
                    bxs = xs[beg_i:end_i, :]
                    bys = ys[beg_i:end_i, :]
                    sess.run(train_step, feed_dict={x_in: bxs, y_label: bys, keep_prob: 0.5})

                # Test
                bxs, bys = self.shuffle_data(train_xs, self.train_ys, 2048)
                rate = sess.run(accuracy, feed_dict={x_in: bxs, y_label: bys, keep_prob: 1.0})
                print("Epoch : %d, training accuracy : %g" % (e + 1, rate))
                if pre_rate < rate:
                    pre_rate = rate
                    saver.save(sess, "params/cnn_net.ckpt")

                if rate < pre_rate / 2:
                    break

            return pre_rate

    def generate_test_data(self, rand_margin, batch_click_num):  #
        print("Generate test data ... ... ")

        self.test_ys = []
        self.test_click_batch = []
        #
        self.cnn_mv_scores = np.empty((0, self.n_classes))  # 单次测试的结果
        self.cnn_mp_scores = np.empty((0, self.n_classes))
        self.cnn_labels = np.empty((0, self.n_classes))
        self.cnn_mv_rates = np.empty((0, self.n_classes))
        self.cnn_mp_rates = np.empty((0, self.n_classes))
        #
        self.gmm_scores = np.empty((0, self.n_classes))
        self.gmm_labels = np.empty((0, self.n_classes))
        self.gmm_rates = np.empty((0, self.n_classes))

        n_classes = self.n_classes
        crop_len = self.crop_len

        for c in range(0, n_classes):
            data = self.test_data[c]
            clicks = data[0].copy()  # 这里必须进行深拷贝,否则浅拷贝后面能量归一化会改变self.test_data中的数据
            count = clicks.shape[0]
            label = np.zeros(n_classes)
            label[c] = 1

            org_len = clicks.shape[1]
            ref_idx = int((org_len - crop_len) / 2 - rand_margin / 2)

            run_num = int(count / batch_click_num)
            if run_num == 0:
                run_num = 1

            for i in range(0, run_num):
                click_batch = np.zeros((batch_click_num, crop_len))
                index = 0
                for j in range(batch_click_num * i, batch_click_num * (i + 1)):
                    r_j = j % count
                    click = clicks[r_j]

                    beg_idx = ref_idx + np.random.randint(0, rand_margin)
                    crop_x = click[beg_idx:(beg_idx + crop_len)]
                    crop_x = np.reshape(crop_x, [1, crop_len])

                    energy = np.sqrt(np.sum(crop_x ** 2))
                    crop_x /= energy

                    click_batch[index, :] = crop_x

                    index += 1

                label = [0] * n_classes
                label[c] = 1

                self.test_ys.append(label)
                self.test_click_batch.append(click_batch)

    def test_cnn(self):

        test_ys = self.test_ys
        click_batchs = copy.deepcopy(self.test_click_batch)

        mv_score = np.zeros((len(test_ys), self.n_classes))
        mp_score = np.zeros((len(test_ys), self.n_classes))
        ref_y = np.zeros((len(test_ys), self.n_classes))
        batch_click_num = click_batchs[0].shape[0]  #

        print("Test cnn ..., batch_click_num = %d ..." % batch_click_num)

        x_in, y_label, keep_prob, y_out, train_step, accuracy, weight = self.cnn_net()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "params/cnn_net.ckpt")  # 加载训练好的网络参数

            for i in range(len(click_batchs)):
                clicks = click_batchs[i]
                xs_batch = self.feature_extractor_cnn(clicks)
                # majority voting
                label_count = np.zeros(self.n_classes)
                net_out_y = sess.run(y_out, feed_dict={x_in: xs_batch, keep_prob: 1.0})
                max_index = np.argmax(net_out_y, axis=1)
                for j in range(len(max_index)):
                    label_count[max_index[j]] += 1

                mv_score[i, :] = label_count/batch_click_num

                # maximum posterior
                net_out_y = sess.run(weight, feed_dict={x_in: xs_batch, keep_prob: 1.0})
                net_out_y = np.sum(net_out_y, axis=0)
                net_out_y = self.soft_max(net_out_y)  # np.exp(net_out_y) / np.sum(np.exp(net_out_y))
                mp_score[i, :] = net_out_y

                # real label
                ref_y[i, :] = np.array(test_ys[i])

        self.cnn_mv_scores = mv_score
        self.cnn_mp_scores = mp_score
        self.cnn_labels = ref_y

    def calculate_rates(self):
        print("\r\nCalculate accuracies ... ...")
        if self.cnn_mv_scores.shape[0] > 0:
            mp_rates = np.zeros((1, self.n_classes))
            mv_rates = np.zeros((1, self.n_classes))
            for c in range(0, self.n_classes):
                index = self.cnn_labels[:, c] == 1
                ref_y = self.cnn_labels[index, :]
                mv_y_score = self.cnn_mv_scores[index, :]
                mp_y_score = self.cnn_mp_scores[index, :]

                prediction = np.equal(np.argmax(mv_y_score, axis=1), np.argmax(ref_y, axis=1))
                prediction.astype(float)
                mv_rates[0, c] = np.mean(prediction)

                prediction = np.equal(np.argmax(mp_y_score, axis=1), np.argmax(ref_y, axis=1))
                prediction.astype(float)
                mp_rates[0, c] = np.mean(prediction)

            print("majority voting")
            print(mv_rates*100)
            print("maximum posterior")
            print(mp_rates*100)
            self.cnn_mv_rates = mv_rates
            self.cnn_mp_rates = mp_rates

        if self.gmm_scores.shape[0] > 0:
            gmm_rates = np.zeros((1, self.n_classes))
            for c in range(0, self.n_classes):
                index = self.gmm_labels[:, c] == 1
                ref_y = self.gmm_labels[index, :]
                gmm_y_score = self.gmm_scores[index, :]

                prediction = np.equal(np.argmax(gmm_y_score, axis=1), np.argmax(ref_y, axis=1))
                prediction.astype(float)
                gmm_rates[0, c] = np.mean(prediction)

            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print("gmm")
            print(gmm_rates * 100)
            self.gmm_rates = gmm_rates

    # load_results加载多次测试的结果数据,本函数统计多次测试识别率的平均值和标准差
    def statics(self):
        print("\r\nCalculate statics ... ...")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        if self.cnn_mv_rates.shape[0] > 0:
            print("majority voting")
            print(np.mean(self.cnn_mv_rates, axis=0) * 100)
            print(np.std(self.cnn_mv_rates, axis=0) * 100)
            print("maximum posterior")
            print(np.mean(self.cnn_mp_rates, axis=0) * 100)
            print(np.std(self.cnn_mp_rates, axis=0) * 100)

        if self.gmm_rates.shape[0] > 0:
            print("gmm")
            print(np.mean(self.gmm_rates, axis=0) * 100)
            print(np.std(self.gmm_rates, axis=0) * 100)

    def draw_metrics(self):

        # The average precision score in multi-label settings
        # For each class

        precision = dict()
        recall = dict()
        average_precision = dict()
        labels = []
        font_size = 12

        precision["micro_mv"], recall["micro_mv"], _ = \
            precision_recall_curve(self.cnn_labels.ravel(), self.cnn_mv_scores.ravel())
        precision["micro_mp"], recall["micro_mp"], _ = \
            precision_recall_curve(self.cnn_labels.ravel(), self.cnn_mp_scores.ravel())
        precision["micro_gmm"], recall["micro_gmm"], _ = \
            precision_recall_curve(self.gmm_labels.ravel(), self.gmm_scores.ravel())

        average_precision["micro_mv"] = average_precision_score(self.cnn_labels, self.cnn_mv_scores, average="micro")
        average_precision["micro_mp"] = average_precision_score(self.cnn_labels, self.cnn_mp_scores, average="micro")
        average_precision["micro_gmm"] = average_precision_score(self.gmm_labels, self.gmm_scores, average="micro")

        labels.append('iso-f1 curves')
        labels.append('CNNMV (area = {0:0.2f})'.format(average_precision["micro_mv"]))
        labels.append('CNNMP (area = {0:0.2f})'.format(average_precision["micro_mp"]))
        labels.append('GMM (area = {0:0.2f})'.format(average_precision["micro_gmm"]))

        # Plot the micro-averaged Precision-Recall curve
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)

        l, = plt.plot(recall['micro_mv'], precision['micro_mv'], 'g-', lw=2)
        lines.append(l)
        l, = plt.plot(recall['micro_mp'], precision['micro_mp'], 'b--', lw=2)
        lines.append(l)
        l, = plt.plot(recall['micro_gmm'], precision['micro_gmm'], 'r-.', lw=2)
        lines.append(l)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.24)
        plt.xlabel('Recall', fontsize=font_size)
        plt.ylabel('Precision', fontsize=font_size)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        # plt.grid()
        # plt.title('Average precision-recall', fontsize=font_size)
        plt.legend(lines, labels, loc=(0, -.32), prop=dict(size=font_size))
        # plt.legend(lines, labels, loc='lower left', fontsize=font_size)

        # ========================================================
        odontocete = self.odontocete
        styles = cycle(['g-', 'b--', 'r-.'])
        for t in range(3):
            precision = dict()
            recall = dict()
            average_precision = dict()
            n_classes = self.n_classes

            if t == 0:
                y_true = self.cnn_labels
                y_score = self.cnn_mv_scores
            elif type == 1:
                y_true = self.cnn_labels
                y_score = self.cnn_mp_scores
            else:
                y_true = self.gmm_labels
                y_score = self.gmm_scores

            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
            average_precision["micro"] = average_precision_score(y_true, y_score, average="micro")

            plt.figure(figsize=(7, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

            lines.append(l)
            labels.append('iso-f1 curves')
            l, = plt.plot(recall["micro"], precision["micro"], 'y:', lw=2)
            lines.append(l)
            labels.append('Average precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

            for i, style in zip(range(n_classes), styles):
                l, = plt.plot(recall[i], precision[i], style, lw=2)
                lines.append(l)
                labels.append('Precision-recall for {0} (area = {1:0.2f})'
                              ''.format(odontocete[i], average_precision[i]))

            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=font_size)
            plt.ylabel('Precision', fontsize=font_size)
            # plt.title('Precision-recall')
            plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=font_size))

        plt.show()


if __name__ == '__main__':

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    n_class = 3
    for t in range(50):
        print('\r\n-------------------------------- Round = %d -------------------------------- \r\n' % (t + 1))
        # classifier = ClickClassifier(ftu_len=128, n_classes=n_class, data_name='W3', odontocete=['Md', 'Gm', 'Gg'])
        # classifier.load_data('./Data/W3_C3npy')
        # classifier.generate_train_data(24, 20000)

        classifier = ClickClassifier(crop_len=256, n_classes=n_class, data_name='W5', odontocete=['Pe', 'Sl', 'Tt'])
        classifier.load_data('./Data/W5_2006_C3npy')
        classifier.generate_train_data(rand_margin=24, n_total=60000)

        while True:
            rate = classifier.train_cnn(36, 256)
            if rate < (1. / n_class + 0.2):
                continue
            else:
                break

        classifier.train_gmm(components=16)

        for b_n in [10, 20, 30, 40, 50]:
            print('\r\n>>>>>>>>>>>>>> batch_click_num = %d' % b_n)
            classifier.generate_test_data(rand_margin=24, batch_click_num=b_n)
            classifier.test_cnn()
            classifier.test_gmm()
            classifier.calculate_rates()
            classifier.save(batch_click_num=b_n)

    # classifier.load_results()
    # classifier.statics()
    # classifier.draw_metrics()
