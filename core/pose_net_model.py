import os
import tensorflow as tf


from utils.tf_utils import in_block, inception_block, auxiliary_classifier_block, conv_block


class PoseNetModel:

    def __init__(self):

        self.weight = 0.3
        self.beta = 300
        self.x = None
        self.y = None
        self.optimizer = None
        self.optimize = None
        self.loss = None
        self.prediction = None

        self.s1 = None
        self.s2 = None

        self.saver = None
    
    def init_saver(self):
        self.saver = tf.train.Saver()


    def save_model(self, sess, path):
        self.saver.save(sess, os.path.join(path, 'pose_net/pose_net_model'))

    def restore_model(self):
        pass

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 7])
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        self.s1 = tf.Variable(0.0)
        self.s2 = tf.Variable(-3.0)


        pred, loss_1, aux_loss_1, aux_loss_2 = self.network()
        self.prediction = pred
        self.loss = loss_1 + self.weight*(aux_loss_1 + aux_loss_2)
        self.optimize = self.minimize(self.loss)

    def network(self):
        feat = in_block(self.x)

        inc_1 = inception_block(feat, f11=64, f11_reduce3=96, f11_reduce5=16, f33=128, f55=32, fpp=32)
        inc_2 = inception_block(inc_1, f11=128, f11_reduce3=128, f11_reduce5=32, f33=192, f55=96, fpp=64)
        inc_3 = inception_block(inc_2, f11=192, f11_reduce3=96, f11_reduce5=16, f33=208, f55=48, fpp=64)

        cl_1 = auxiliary_classifier_block(inc_3)

        inc_4 = inception_block(inc_3, f11=160, f11_reduce3=112, f11_reduce5=24, f33=224, f55=64, fpp=64)
        inc_5 = inception_block(inc_4, f11=128, f11_reduce3=128, f11_reduce5=32, f33=256, f55=64, fpp=64)
        inc_6 = inception_block(inc_5, f11=112, f11_reduce3=144, f11_reduce5=32, f33=288, f55=64, fpp=64)

        cl_2 = auxiliary_classifier_block(inc_6)

        inc_7 = inception_block(inc_6, f11=256, f11_reduce3=160, f11_reduce5=32, f33=320, f55=128, fpp=64)
        inc_8 = inception_block(inc_7, f11=256, f11_reduce3=160, f11_reduce5=32, f33=320, f55=128, fpp=64)
        # inc_9 = inception_block(inc_8, f11=384, f11_reduce3=192, f11_reduce5=48, f33=384, f55=128, fpp=64)
        # inc_9 = inception_block(inc_8, f11=256, f11_reduce3=160, f11_reduce5=48, f33=320, f55=128, fpp=64)
        inc_9 = inc_8

        av_pool = tf.nn.avg_pool(inc_9, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')

        # av_pool = conv_block(av_pool, k_size=1, c_in=960, c_out=640)

        fc_1 = tf.layers.dense(tf.layers.flatten(av_pool), units=2048)
        
        fc_pos = tf.layers.dense(fc_1, units=3)
        fc_qua = tf.layers.dense(fc_1, units=4)
        
        fc_2 = tf.concat([fc_pos, fc_qua], 1)

        return fc_2, self.get_loss(fc_2, self.y), self.get_loss(cl_1, self.y), self.get_loss(cl_2, self.y)

    def get_loss(self, est, true):
        # est[:, 3:] = est[:, 3:]/tf.norm(est[:, 3:])
        pos, qua = tf.split(est, [3, 4], 1)
        qua = tf.div(qua, tf.sqrt(tf.reduce_sum(tf.square(qua), axis=1)))
        return tf.exp(-self.s1)*tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(est[:, :3], true[:, :3])), axis=1)) + \
               tf.exp(-self.s2)*tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(qua, true[:, 3:])), axis=1))) + self.s1 + self.s2

    def minimize(self, loss):

        return self.optimizer.minimize(loss)
