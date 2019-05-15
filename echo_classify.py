import os
import numpy as np
import find_click
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class EchoCnnClassifier:
    def __init__(self, ftu_num, n_class):
        self.ftu_num = ftu_num
        self.n_class = n_class
        self.train_xs = np.empty((0, self.ftu_num))
        self.train_ys = np.empty((0, self.n_class))
        self.test_xs = np.empty((0, self.ftu_num))
        self.test_ys = np.empty((0, self.n_class))

    @staticmethod
    def down_sample(data, step):
        out_data = []
        for i in range(0, data.shape[0]):
            if i % step == 0:
                out_data.append(data[i])
        return out_data

    @staticmethod
    def random_crop_data(data_in, start, scope, length, snr):
        beg_idx = start + np.random.randint(0, scope)
        crop_x = data_in[beg_idx:(beg_idx + length)]
        noise = np.random.rand(length)
        energy_noise = np.sum(np.multiply(np.array(noise), np.array(noise)))
        energy_data = np.sum(np.multiply(np.array(crop_x), np.array(crop_x)))
        scale = np.sqrt(energy_data / energy_noise / (10 ** (snr / 10)))

        noise = [v * scale for v in noise]
        crop_x = [crop_x[i] + noise[i] for i in range(min(len(noise), len(crop_x)))]

        return crop_x

    @staticmethod
    def split_data(xs, ratio=0.5):
        num = len(xs)
        split_idx = int(num * ratio)
        xs0 = xs[0:split_idx]
        xs1 = xs[split_idx:]
        return xs0, xs1

    def generate_data(self, samples, n_total):
        num = len(samples)
        xs = []

        for i in range(0, n_total):
            index = i % num
            sample = samples[index]
            snr = np.random.randint(5, 10)
            sample = self.random_crop_data(sample, 32, 96, 512, snr)
            xs.append(sample)
        return xs

    @staticmethod
    def plot_data(file_path):
        cvs_data = pd.read_csv(file_path, sep='\t')
        data = cvs_data.values
        sample = data[:, 1]
        sample = EchoCnnClassifier.down_sample(sample, 2)
        sample = EchoCnnClassifier.random_crop_data(sample, 32, 96, 512, 0)

        print(data.shape)
        plt.figure()
        plt.plot(sample)
        plt.xlabel("Time(us)")
        plt.ylabel("Amplitude")
        plt.grid('True')  # 标尺，on：有，off:无。
        plt.show()

    def load_data(self, data_path, n_total=20000):
        self.train_xs = np.empty((0, self.ftu_num))
        self.train_ys = np.empty((0, self.n_class))
        self.test_xs = np.empty((0, self.ftu_num))
        self.test_ys = np.empty((0, self.n_class))

        for c in range(0, self.n_class):
            path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
            files = find_click.list_files(path, '.txt')

            print("load data : %s, the number of files : %d" % (path, len(files)))

            label = np.zeros(self.n_class)
            label[c] = 1

            samples = []
            for file in files:
                cvs_data = pd.read_csv(file, sep='\t')
                data = cvs_data.values
                sample = data[:, 1]
                sample = self.down_sample(sample, 2)
                samples.append(sample)

            xs0, xs1 = self.split_data(samples)

            xs0 = self.generate_data(xs0, int(n_total * 4 / 5))
            xs1 = self.generate_data(xs1, int(n_total / 5))

            xs0 = np.array(xs0)
            xs1 = np.array(xs1)

            ys0 = np.tile(label, (xs0.shape[0], 1))
            ys1 = np.tile(label, (xs1.shape[0], 1))

            self.train_xs = np.vstack((self.train_xs, xs0))
            self.train_ys = np.vstack((self.train_ys, ys0))
            self.test_xs = np.vstack((self.test_xs, xs1))
            self.test_ys = np.vstack((self.test_ys, ys1))

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
    def shuffle_lists(xs, ys, num):
        shape = xs.shape
        ri = np.random.permutation(shape[0])
        ri = ri[0: num]
        batch_xs = np.empty((0, xs.shape[1]))
        batch_ys = np.empty((0, ys.shape[1]))
        for i in ri:
            batch_xs = np.vstack((batch_xs, xs[i]))
            batch_ys = np.vstack((batch_ys, ys[i]))

        return batch_xs, batch_ys

    def train_cnn(self):

        print("train cnn ... ...")

        print(self.train_xs.shape)
        print(self.test_xs.shape)

        x = tf.placeholder("float", [None, self.ftu_num])
        y_ = tf.placeholder("float", [None, self.n_class])

        # 输入
        x_image = tf.reshape(x, [-1, 1, self.ftu_num, 1])

        # 第一个卷积层
        W_conv1 = self.weight_variable([1, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_1x2(h_conv1)

        # 第二个卷积层
        W_conv2 = self.weight_variable([1, 5, 32, 16])
        b_conv2 = self.bias_variable([32])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_1x2(h_conv2)

        # 密集链接层
        W_fc1 = self.weight_variable([1 * (self.ftu_num/4) * 16, 64])
        b_fc1 = self.bias_variable([64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * (self.ftu_num/4) * 16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

        # 输出层
        W_fc2 = self.weight_variable([64, self.n_class])
        b_fc2 = self.bias_variable([self.n_class])
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

        # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        init = tf.global_variables_initializer()

        # saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(20000):
                bxs, bys = self.shuffle_lists(self.train_xs, self.train_ys, 160)
                if (i + 1) % 1000 == 0:
                    print("step : %d, training accuracy : %g" %
                          (i + 1, sess.run(accuracy, feed_dict={x: bxs, y_: bys, keep_prob: 1.0})))

                sess.run(train_step, feed_dict={x: bxs, y_: bys, keep_prob: 0.5})

            # saver.save(sess, "params/cnn_net.ckpt")

            # print("test accuracy : %g" % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})))
            sample_num = self.test_xs.shape[0]

            print("test cnn ... ...")
            correct_count = 0
            for j in range(0, sample_num):
                txs = self.test_xs[j]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                if np.equal(np.argmax(out_y), np.argmax(self.test_ys[j])):
                    correct_count += 1

            print('test accuracy: ', round(correct_count / sample_num, 3))


if __name__ == '__main__':

    # EchoCnnClassifier.plot_data('steel_echo_pure.txt')
    classifier = EchoCnnClassifier(512, 2)
    classifier.load_data('./echo', 100)
    classifier.train_cnn()
