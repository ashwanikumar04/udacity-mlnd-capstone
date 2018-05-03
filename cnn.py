import tensorflow as tf
from helpers import one_hot_encode, get_batch, get_training_set, get_test_set, log
from sklearn.utils import shuffle


class CNN:
    def __init__(self, params, labels, image_size):
        self.params = params
        self.labels = labels
        self.image_size = image_size

    def init_weights(self, shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(self, x, W):
        # x --> [batch,H,W,channels]
        # W --> [filter H, filter W, Channels In, Channels Out]
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x):
        # x --> [batch,H,W,channels]
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def convolutional_layer(self, input_x, shape):
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)

    def normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b

    def run(self, train_X, train_y, test_X, test_y, validate_X, validate_y):
        x = tf.placeholder(tf.float32, shape=[None, self.image_size])
        y_true = tf.placeholder(tf.float32, shape=[None, self.labels])
        x_image = tf.reshape(x, [-1, 48, 48, 1])
        convo_1 = self.convolutional_layer(x_image, shape=[5, 5, 1, 32])
        convo_1_pooling = self.max_pool_2x2(convo_1)

        convo_2 = self.convolutional_layer(
            convo_1_pooling, shape=[5, 5, 32, 64])
        convo_2_pooling = self.max_pool_2x2(convo_2)

        # Why 12 by 12 image? Because we did 2 pooling layers, so (48/2)/2 = 12
        convo_2_flat = tf.reshape(convo_2_pooling, [-1, 12 * 12 * 64])
        full_layer_1 = tf.nn.relu(self.normal_full_layer(convo_2_flat, 1024))

        hold_prob = tf.placeholder(tf.float32, name="hold_prob")
        full_one_dropout = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)

        y_pred = self.normal_full_layer(full_one_dropout, self.labels)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.learning_rate)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        correct_prediction = tf.equal(
            tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(init)
            for step in range(self.params.epoch):
                X, y = shuffle(train_X, train_y)
                for current_batch in range(self.params.num_batches):
                    batch_X, batch_y = get_batch(
                        X, y, current_batch, self.params.batch_size)
                    sess.run(
                        train, feed_dict={
                            x: batch_X,
                            y_true: batch_y,
                            hold_prob: self.params.hold_prob
                        })
                if step % self.params.epoch_to_report == 0:
                    log(step, "Epoch")
                    log(
                        sess.run(
                            accuracy,
                            feed_dict={
                                x: test_X,
                                y_true: test_y,
                                hold_prob: 1.0
                            }), "model accuracy")
            log(
                sess.run(
                    accuracy,
                    feed_dict={
                        x: validate_X,
                        y_true: validate_y,
                        hold_prob: 1.0
                    }), "Final accuracy")
