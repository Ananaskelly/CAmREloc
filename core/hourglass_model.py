import os
import tensorflow as tf

from utils.tf_utils import residual_block, hourglass_regressor, conv_block, decoder_block


class HourglassModel:
    def __init__(self):
        self.x = None
        self.y = None
        self.optimizer = None
        self.optimize = None
        self.loss = None
        self.prediction = None

        self.s1 = None
        self.s2 = None

    def init_saver(self):
        self.saver = tf.train.Saver()

    def save_model(self, sess, path):
        self.saver.save(sess, os.path.join(path, 'hourglass/hourglass_model'))

    def restore_model(self):
        pass

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 7])
        self.phase = tf.placeholder(tf.bool)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.s1 = tf.Variable(0.0)
        self.s2 = tf.Variable(-3.0)

        pred, self.loss = self.network()

        self.prediction = pred
        self.optimize = self.minimize(self.loss)

    def network(self):
        conv_1 = conv_block(self.x, k_size=7, c_in=3, c_out=64, strides=[1, 2, 2, 1])

        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        last_feat = []

        feat = pool_1
        for i in range(3):
            feat = residual_block(feat, filter_size=3, c_out=64, is_train=self.phase)

        last_feat.append(feat)
        feat = residual_block(feat, filter_size=3, c_out=128, enable_proj_shortcut=True,
                              strides=[1, 2, 2, 1], is_train=self.phase)

        for i in range(3):
            feat = residual_block(feat, filter_size=3, c_out=128, is_train=self.phase)
        last_feat.append(feat)
        feat = residual_block(feat, filter_size=3, c_out=256, enable_proj_shortcut=True,
                              strides=[1, 2, 2, 1], is_train=self.phase)

        for i in range(5):
            feat = residual_block(feat, filter_size=3, c_out=256, is_train=self.phase)

        last_feat.append(feat)
        feat = residual_block(feat, filter_size=3, c_out=512, enable_proj_shortcut=True,
                              strides=[1, 2, 2, 1], is_train=self.phase)

        for i in range(2):
            feat = residual_block(feat, filter_size=3, c_out=512, is_train=self.phase)

        up_conv_1 = decoder_block(feat, 4, 256, out_size=14, is_train=self.phase) + last_feat[2]
        up_conv_2 = decoder_block(up_conv_1, 4, 128, out_size=28, is_train=self.phase) + last_feat[1]
        up_conv_3 = decoder_block(up_conv_2, 4, 64, out_size=56, is_train=self.phase) + last_feat[0]

        conv_final = conv_block(up_conv_3, k_size=3, c_in=64, c_out=32)
        conv_final = tf.layers.batch_normalization(conv_final, training=self.phase)

        out_final = hourglass_regressor(conv_final)

        return out_final, self.get_loss(out_final, self.y)

    def get_loss(self, est, true):
        # est[:, 3:] = est[:, 3:]/tf.norm(est[:, 3:])
        pos, qua = tf.split(est, [3, 4], 1)
        qua = tf.div(qua, tf.sqrt(tf.reduce_sum(tf.square(qua), axis=1)))
        return tf.exp(-self.s1) * tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(est[:, :3], true[:, :3])), axis=1)) + \
            tf.exp(-self.s2) * tf.sqrt(
                tf.reduce_sum(tf.square(tf.subtract(qua, true[:, 3:])), axis=1))) + self.s1 + self.s2

    def minimize(self, loss):
        return self.optimizer.minimize(loss)
