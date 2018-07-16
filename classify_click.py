import tensorflow as tf
import find_click
import wave
import numpy as np


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


x = tf.placeholder("float", [None, 192])
y_ = tf.placeholder("float", [None, 3])

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
W_fc2 = weight_variable([256, 3])
b_fc2 = bias_variable([3])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


def load_data():

    n_total = 20000
    train_xs = np.empty((0, 192))
    train_ys = np.empty((0, 3))
    test_xs = np.empty((0, 192))
    test_ys = np.empty((0, 3))

    for c in range(0, 3):
        path = "./Data/Click/%(class)d" % {'class': c}
        wav_files = find_click.list_wav_files(path)

        temp_train_xs = np.empty((0, 256))
        temp_train_ys = np.empty((0, 3))
        temp_test_xs = np.empty((0, 256))
        temp_test_ys = np.empty((0, 3))

        count = 0
        for pathname in wav_files:
            count += 1

            wave_data, frameRate = find_click.read_wav_file(pathname)

            energy = np.sqrt(np.sum(wave_data ** 2))
            wave_data /= energy
            wave_data = np.reshape(wave_data, [-1])

            label = np.array([0, 0, 0])
            label[c] = 1

            if count % 5 == 0:
                temp_test_xs = np.vstack((temp_test_xs, wave_data))
                temp_test_ys = np.vstack((temp_test_ys, label))
            else:
                temp_train_xs = np.vstack((temp_train_xs, wave_data))
                temp_train_ys = np.vstack((temp_train_ys, label))

        n_re = np.int32(n_total/count)
        shape = temp_train_xs.shape
        for i in range(0, shape[0]):
            xs = temp_train_xs[i]
            for j in range(0, n_re):
                beg_idx = np.random.randint(0, 32)
                new_xs = xs[beg_idx:(beg_idx+192)]
                train_xs = np.vstack((train_xs, new_xs))
                train_ys = np.vstack((train_ys, temp_train_ys[i]))

        shape = temp_test_xs.shape
        for i in range(0, shape[0]):
            xs = temp_test_xs[i]
            for j in range(0, n_re):
                beg_idx = np.random.randint(0, 32)
                new_xs = xs[beg_idx:(beg_idx+192)]
                test_xs = np.vstack((test_xs, new_xs))
                test_ys = np.vstack((test_ys, temp_test_ys[i]))

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


train_xs, train_ys, test_xs, test_ys = load_data()

print(train_xs.shape)
print(test_xs.shape)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        bxs, bys = shufflelists(train_xs, train_ys, 100)
        if (i + 1) % 1000 == 0:
            print("step : %d, training accuracy : %g" % (i, sess.run(accuracy, feed_dict={x: bxs, y_: bys, keep_prob: 1.0})))

        sess.run(train_step, feed_dict={x: bxs, y_: bys, keep_prob: 0.5})
        # train_step.run(feed_dict={x: bxs, y_: bys, keep_prob: 0.5})

    print("test accuracy : %g" % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})))

