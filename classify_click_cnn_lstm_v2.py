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


def split_data(xs):
    num = xs.shape[0]
    split_idx = int(num * 4/5)
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

        temp_train_xs = random_crop(xs0, batch_num, int(n_total*4/5))
        temp_test_xs = random_crop(xs1, batch_num, int(n_total/5))

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
    print("train cnn ... ...")

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
                print("step : %d, training accuracy : %g" % (i+1, sess.run(accuracy, feed_dict={x: bxs, y_: bys, keep_prob: 1.0})))

            sess.run(train_step, feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
            # train_step.run(feed_dict={x: bxs, y_: bys, keep_prob: 0.5})

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


# 将CNN最后一层全连接层的输出作为LSTM的输入特征
def load_data_lstm(data_path, n_class, batch_num=20, n_total=500):

    train = []
    test = []

    x_in = tf.placeholder("float", [None, 192])

    # 输入
    x_image = tf.reshape(x_in, [-1, 1, 192, 1])

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

        test_cnn = []

        for c in range(0, n_class):
            # path = "./Data/Click/%(class)d" % {'class': c}
            path = "%(path)s/%(class)d" % {'path': data_path, 'class': c}
            wav_files = find_click.list_wav_files(path)

            print("load data : %s, the number of files : %d" % (path, len(wav_files)))

            # xs = np.empty((0, 256))
            xs = np.empty((0, 320))
            count = 0
            for pathname in wav_files:
                wave_data, frame_rate = find_click.read_wav_file(pathname)
                energy = np.sqrt(np.sum(wave_data ** 2))
                wave_data /= energy
                wave_data = np.reshape(wave_data, [-1])
                xs = np.vstack((xs, wave_data))
                count += 1
                if count > batch_num * n_total:
                    break

            xs0, xs1 = split_data(xs)

            sample_num = xs0.shape[0]
            for i in range(0, int(n_total * 4/5)):
                frames = np.empty((0, 256))
                for j in range(batch_num * i, batch_num * (i + 1)):
                    index = j % sample_num
                    temp_x = xs0[index]
                    # beg_idx = np.random.randint(0, 32)
                    beg_idx = np.random.randint(64, (64+32))
                    crop_x = temp_x[beg_idx:(beg_idx + 192)]
                    crop_x = np.reshape(crop_x, [1, 192])
                    ftu = sess.run(h_fc1, feed_dict={x_in: crop_x})  # 计算CNN网络输出
                    frames = np.vstack((frames, ftu))

                frames = np.expand_dims(np.expand_dims(frames, axis=0), axis=0)
                frames = list(frames)

                label = [0] * n_class
                label[c] = 1
                label = np.array([[label]])
                label = list(label)
                sample = frames + label
                train.append(sample)

            sample_num = xs1.shape[0]
            for i in range(0, int(n_total/5)):
                frames = np.empty((0, 256))
                tmp_xs = np.empty((0, 192))
                for j in range(batch_num * i, batch_num * (i + 1)):
                    index = j % sample_num
                    temp_x = xs1[index]
                    # beg_idx = np.random.randint(0, 32)
                    beg_idx = np.random.randint(64, (64+32))
                    crop_x = temp_x[beg_idx:(beg_idx + 192)]
                    crop_x = np.reshape(crop_x, [1, 192])
                    ftu = sess.run(h_fc1, feed_dict={x_in: crop_x})  # 计算CNN网络输出
                    frames = np.vstack((frames, ftu))
                    tmp_xs = np.vstack((tmp_xs, crop_x))

                frames = np.expand_dims(np.expand_dims(frames, axis=0), axis=0)
                frames = list(frames)

                label = [0] * n_class
                label[c] = 1
                label = np.array([[label]])
                label = list(label)
                sample = frames + label

                test.append(sample)

                tmp_xs = np.expand_dims(np.expand_dims(tmp_xs, axis=0), axis=0)
                tmp_xs = list(tmp_xs)
                sample = tmp_xs + label
                test_cnn.append(sample)

        count = 0
        for i in range(len(test_cnn)):

            test_xs = test_cnn[i][0]

            label = np.zeros(n_class)
            for j in range(0, test_xs.shape[1]):
                txs = test_xs[0, j, :]
                txs = np.reshape(txs, [1, 192])
                out_y = sess.run(y, feed_dict={x_in: txs, keep_prob: 1.0})
                c = np.argmax(out_y, 1)
                label[c] += 1

            ref_y = test_cnn[i][1]
            if np.equal(np.argmax(label), np.argmax(ref_y)):
                count += 1

        print('cnn test accuracy: ', round(count / len(test_cnn), 3))

    return train, test


# LSTM
def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)


def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


def orthogonal_initializer(shape, scale=1.0):
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]], trainable=True, dtype=tf.float32)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
        # var
        # the shape of incoming is [n_samples, n_steps, D_cell]
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        # parameters
        # igate = W_xi.* x + W_hi.* h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))
        # fgate = W_xf.* x + W_hf.* h + b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # ogate = W_xo.* x + W_ho.* h + b_o
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros([self.D_cell]))
        # cell = W_xc.* x + W_hc.* h + b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c = tf.Variable(tf.zeros([self.D_cell]))

        # init cell and hidden state whose shapes are [n_samples, D_cell]
        init_for_both = tf.matmul(self.incoming[:, 0, :], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # because tf.scan only takes two arguments, the hidden state and cell are needed to merge
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # transpose the tensor so that the first dim is time_step
        self.incoming = tf.transpose(self.incoming, perm=[1, 0, 2])

    def one_step(self, previous_h_c_tuple, current_x):
        # to split hidden state and cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)

        # computing
        # input gate
        i = tf.sigmoid(
            tf.matmul(current_x, self.W_xi) +
            tf.matmul(prev_h, self.W_hi) +
            self.b_i)
        # forget Gate
        f = tf.sigmoid(
            tf.matmul(current_x, self.W_xf) +
            tf.matmul(prev_h, self.W_hf) +
            self.b_f)
        # output Gate
        o = tf.sigmoid(
            tf.matmul(current_x, self.W_xo) +
            tf.matmul(prev_h, self.W_ho) +
            self.b_o)
        # new cell info
        c = tf.tanh(
            tf.matmul(current_x, self.W_xc) +
            tf.matmul(prev_h, self.W_hc) +
            self.b_c)
        # current cell
        current_c = f * prev_c + i * c
        # current hidden state
        current_h = o * tf.tanh(current_c)

        return tf.stack([current_h, current_c])

    def all_steps(self):
        # inputs shape : [n_sample, n_steps, D_input]
        # outputs shape : [n_steps, n_sample, D_output]
        hstates = tf.scan(fn=self.one_step,
                          elems=self.incoming,
                          initializer=self.previous_h_c_tuple,
                          name='hstates')[:, 0, :, :]
        return hstates


def shuffle_frames(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


n_class = 8
# train_cnn('./Data/Click', 3, 20, 200)
# train_cnn('./Data/ClickC8', n_class, 20, 500)
# exit()

train, test = load_data_lstm('./Data/ClickC8', n_class, 20, 1000)

print("train lstm ... ...")
print('num of train sequences:%s' % len(train))  #
print('num of test sequences:%s' % len(test))    #
print('shape of inputs:', test[0][0].shape)     # (1,n,256)
print('shape of labels:', test[0][1].shape)     # (1,3)


D_input = 256
D_label = n_class
learning_rate = 7e-5
num_units = 512

inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")

rnn_cell = LSTMcell(inputs, D_input, num_units, orthogonal_initializer)
rnn_out = rnn_cell.all_steps()
# reshape for output layer
rnn = tf.reshape(rnn_out, [-1, num_units])
# output layer
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.nn.softmax(tf.matmul(rnn, W) + b)

loss = -tf.reduce_mean(labels * tf.log(output))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# 训练并记录
def train_epoch(epoch):
    for k in range(epoch):
        train_shift = shuffle_frames(train)
        for i in range(len(train)):
            sess.run(train_step, feed_dict={inputs: train_shift[i][0], labels: train_shift[i][1]})
        tl = 0
        dl = 0
        for i in range(len(test)):
            dl += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1]})
        for i in range(len(train)):
            tl += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1]})

        print(k, 'train:', round(tl / len(train), 3), '  test:', round(dl / len(test), 3))
        count = 0
        for j in range(len(test)):
            pred = sess.run(output, feed_dict={inputs: test[j][0]})
            pred_len = len(pred)
            max_pred = list(pred[pred_len - 1]).index(max(list(pred[pred_len - 1])))
            max_test = list(test[j][1][0]).index(max(list(test[j][1][0])))
            if max_pred == max_test:
                count += 1
        print('test accuracy: ', round(count / len(test), 3))


t0 = time.time()
train_epoch(10)
t1 = time.time()
print(" %f seconds" % round((t1 - t0), 2))