import tensorflow as tf
import find_click
import numpy as np
import time


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def split_data(xs):
    num = xs.shape[0]
    split_idx = int(num * 4 / 5)
    xs0 = xs[0:split_idx, :]
    xs1 = xs[split_idx:num, :]
    return xs0, xs1


def random_crop(xs, batch_num, n_total):
    num = xs.shape[0]
    rc_xs = np.empty((0, 192))

    for i in range(0, n_total):
        for j in range(batch_num * i, batch_num * (i + 1)):
            index = j % num
            temp_x = xs[index]
            # beg_idx = np.random.randint(0, 32)
            beg_idx = np.random.randint(64, (64 + 32))
            crop_x = temp_x[beg_idx:(beg_idx + 192)]
            crop_x = np.reshape(crop_x, [1, 192])
            rc_xs = np.vstack((rc_xs, crop_x))

    return rc_xs


def load_data(data_path, n_class, batch_num=20, n_total=500):
    train_xs = np.empty((0, 192))
    train_ys = np.empty((0, n_class))
    test_xs = np.empty((0, 192))
    test_ys = np.empty((0, n_class))

    for c in range(0, n_class):
        path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
        wav_files = find_click.list_wav_files(path)

        print("load data : %s, the number of files : %d" % (path, len(wav_files)))

        label = np.zeros(n_class)
        label[c] = 1

        # xs = np.empty((0, 256))
        xs = np.empty((0, 320))
        count = 0
        #
        for pathname in wav_files:
            wave_data, frame_rate = find_click.read_wav_file(pathname)

            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])
            xs = np.vstack((xs, wave_data))
            count += 1
            if count >= batch_num * n_total:
                break

        xs0, xs1 = split_data(xs)

        temp_train_xs = random_crop(xs0, batch_num, int(n_total * 4 / 5))
        temp_test_xs = random_crop(xs1, batch_num, int(n_total / 5))

        temp_train_ys = np.tile(label, (temp_train_xs.shape[0], 1))
        temp_test_ys = np.tile(label, (temp_test_xs.shape[0], 1))

        train_xs = np.vstack((train_xs, temp_train_xs))
        train_ys = np.vstack((train_ys, temp_train_ys))
        test_xs = np.vstack((test_xs, temp_test_xs))
        test_ys = np.vstack((test_ys, temp_test_ys))

    return train_xs, train_ys, test_xs, test_ys


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


def train_cnn(data_path, n_class, batch_num=20, n_total=500):

    print("train cnn for one click ... ...")

    train_xs, train_ys, test_xs, test_ys = load_data(data_path, n_class, batch_num, n_total)

    print(train_xs.shape)
    print(test_xs.shape)

    x = tf.placeholder("float", [None, 192])
    y_ = tf.placeholder("float", [None, n_class])

    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

    # 第一个卷积层
    W_conv1 = weight_variable([1, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)

    # 第二个卷积层
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    # 密集链接层
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(20000):
            bxs, bys = shufflelists(train_xs, train_ys, 160)
            if (i + 1) % 1000 == 0:
                print("step : %d, training accuracy : %g" %
                      (i + 1, sess.run(accuracy, feed_dict={x: bxs, y_: bys, keep_prob: 1.0})))

            sess.run(train_step, feed_dict={x: bxs, y_: bys, keep_prob: 0.5})

        saver.save(sess, "params/cnn_net.ckpt")

        # print("test accuracy : %g" % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})))
        sample_num = test_xs.shape[0]

        correct_cout = 0
        for j in range(0, sample_num):
            txs = test_xs[j]
            txs = np.reshape(txs, [1, 192])
            out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
            if np.equal(np.argmax(out_y), np.argmax(test_ys[j])):
                correct_cout += 1

        print('test accuracy: ', round(correct_cout / sample_num, 3))

        batch_index = 0
        test_cout = 0
        correct_cout = 0

        while (True):
            if batch_num * (batch_index + 1) > sample_num:
                break

            test_cout += 1
            label = np.zeros(n_class)
            for j in range(batch_num * batch_index, batch_num * (batch_index + 1)):
                txs = test_xs[j]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            sample_index = batch_num * batch_index
            ref_y = test_ys[sample_index]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                correct_cout += 1

            batch_index += 1

        print('batch test accuracy: ', round(correct_cout / test_cout, 3))


#
def test_cnn_batch_data(data_path, n_class, batch_num=20):

    x = tf.placeholder("float", [None, 192])
    # 输入
    x_image = tf.reshape(x, [-1, 1, 192, 1])

    # 第一个卷积层
    W_conv1 = weight_variable([1, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_1x2(h_conv1)

    # 第二个卷积层
    W_conv2 = weight_variable([1, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)

    # 密集链接层
    W_fc1 = weight_variable([1 * 48 * 32, 256])
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 48 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # 输出层
    W_fc2 = weight_variable([256, n_class])
    b_fc2 = bias_variable([n_class])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "params/cnn_net.ckpt")  # 加载训练好的网络参数

        for c in range(0, n_class):
            path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}

            npy_files = find_click.list_files(path, '.npy')
            print("load data : %s, the number of files : %d" % (path, len(npy_files)))

            for path_name in npy_files:
                print(path_name)
                clicks = np.load(path_name)

                print("the number of clicks : %d" % (clicks.shape[0]))

                click_batch = []

                num = clicks.shape[0]
                for i in range(0, int(num / batch_num)):
                    tmp_xs = np.empty((0, 192))
                    for j in range(batch_num * i, batch_num * (i + 1)):
                        index = j % num
                        temp_x = clicks[index]

                        energy = np.sqrt(np.sum(temp_x ** 2))
                        temp_x /= energy

                        beg_idx = np.random.randint(64, (64 + 32))
                        crop_x = temp_x[beg_idx:(beg_idx + 192)]
                        crop_x = np.reshape(crop_x, [1, 192])
                        tmp_xs = np.vstack((tmp_xs, crop_x))

                    label = [0] * n_class
                    label[c] = 1

                    label = np.array([[label]])
                    label = list(label)

                    tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
                    tmp_xs = list(tmp_xs)
                    sample = tmp_xs + label
                    click_batch.append(sample)

                count = 0
                for i in range(len(click_batch)):
                    temp_xs = click_batch[i][0]
                    label = np.zeros(n_class)
                    for j in range(0, temp_xs.shape[1]):
                        txs = temp_xs[0, j, :]
                        txs = np.reshape(txs, [1, 192])
                        out_y = sess.run(y, feed_dict={x: txs, keep_prob: 1.0})
                        max_idx = np.argmax(out_y, 1)
                        label[max_idx] += 1

                    ref_y = click_batch[i][1]
                    if np.equal(np.argmax(label), np.argmax(ref_y)):
                        count += 1

                if len(click_batch) == 0:
                    continue

                print('cnn test accuracy (majority voting): ', round(count / len(click_batch), 3))

                count = 0
                weight = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                for i in range(len(click_batch)):
                    temp_xs = click_batch[i][0]
                    label = np.zeros(n_class)
                    for j in range(0, temp_xs.shape[1]):
                        txs = temp_xs[0, j, :]
                        txs = np.reshape(txs, [1, 192])
                        out = sess.run(weight, feed_dict={x: txs, keep_prob: 1.0})
                        out = np.reshape(out, label.shape)
                        label = label + out

                    ref_y = click_batch[i][1]
                    if np.equal(np.argmax(label), np.argmax(ref_y)):
                        count += 1

                print('cnn test accuracy (weight voting): ', round(count / len(click_batch), 3))


batch_num = 10
n_class = 8

# train_cnn('./Data/Click', 3, 20, 200)
# train_cnn('./Data/ClickC8', n_class, 20, 500)
# exit()

# test_cnn_batch_data('./Data/ClickC8npy', n_class, batch_num)



