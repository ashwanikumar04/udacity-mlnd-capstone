import tensorflow as tf
from helpers import one_hot_encode, get_batch, get_training_set, get_test_set, log
from sklearn.utils import shuffle

class LinearClassifer:
    def __init__(self, params, labels, image_size):
        self.params = params
        self.labels = labels
        self.image_size = image_size

    def run(self, train_X, train_y, test_X, test_y, validate_X, validate_y):
        accuracyDictionary = {};
        x = tf.placeholder(tf.float32, shape=[None, self.image_size])
        W = tf.Variable(tf.zeros([self.image_size, self.labels]))
        b = tf.Variable(tf.zeros([self.labels]))
        y = tf.matmul(x, W) + b
        y_true = tf.placeholder(tf.float32, [None, self.labels])
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.params.learning_rate)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(init)
            for step in range(self.params.epoch):
                X, y = shuffle(train_X, train_y)
                for current_batch in range(self.params.num_batches):
                    batch_X, batch_y = get_batch(
                        X, y, current_batch, self.params.batch_size)
                    sess.run(train, feed_dict={x: batch_X, y_true: batch_y})
                if step % self.params.epoch_to_report == 0:
                    log(step,"Epoch")
                    accuracyDictionary[step]=sess.run(accuracy, feed_dict={x: test_X,
                                                      y_true: test_y})
                    log(accuracyDictionary[step], "model accuracy")

            log(sess.run(accuracy, feed_dict={x: validate_X,
                                              y_true: validate_y}), "Final accuracy")
        return accuracyDictionary
