import tensorflow as tf


from utils.tf_utils import in_block, inception_block, auxiliary_classifier_block


class PoseNetModel:

    def __init__(self):

        self.weight = 0.3
        self.beta = 500
        self.x = None
        self.y = None
        self.optimizer = None
        self.optimize = None

    def save_model(self):
        pass

    def restore_model(self):
        pass

    def build_model(self):
        self.x = tf.placeholder()
        self.y = tf.placeholder()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0)

        pred, loss_1, aux_loss_1, aux_loss_2 = self.network()
        final_loss = loss_1 + self.weight*(aux_loss_1 + aux_loss_2)
        self.optimize = self.minimize(final_loss)

    def network(self):
        feat = in_block(self.x)

        inc_1 = inception_block(feat, f11=64, f11_reduce3=96, f11_reduce5=16, f33=128, f55=32)
        inc_2 = inception_block(inc_1, f11=128, f11_reduce3=128, f11_reduce5=32, f33=192, f55=96)
        inc_3 = inception_block(inc_2, f11=192, f11_reduce3=96, f11_reduce5=16, f33=208, f55=48)

        cl_1 = auxiliary_classifier_block(inc_3)

        inc_4 = inception_block(inc_3, f11=160, f11_reduce3=112, f11_reduce5=24, f33=224, f55=64)
        inc_5 = inception_block(inc_4, f11=128, f11_reduce3=128, f11_reduce5=32, f33=256, f55=64)
        inc_6 = inception_block(inc_5, f11=112, f11_reduce3=144, f11_reduce5=32, f33=288, f55=64)

        cl_2 = auxiliary_classifier_block(inc_6)

        inc_7 = inception_block(inc_6, f11=256, f11_reduce3=160, f11_reduce5=32, f33=320, f55=128)
        inc_8 = inception_block(inc_7, f11=256, f11_reduce3=160, f11_reduce5=32, f33=320, f55=128)
        inc_9 = inception_block(inc_8, f11=384, f11_reduce3=192, f11_reduce5=48, f33=384, f55=128)

        av_pool = tf.nn.avg_pool(inc_9, ksize=7, strides=[1, 1])
        fc_1 = tf.layers.dense(tf.layers.flatten(av_pool), units=0)
        fc_2 = tf.layers.dense(fc_1, units=0)

        return fc_2, self.loss(fc_2, self.y), self.loss(cl_1, self.y), self.loss(cl_2, self.y)

    def loss(self, est, true):

        return tf.norm(true[:, :3] - est[:, :3], ord=2) + self.beta*tf.norm(true[:, 3:] -
                                                                            est[:, 3:]/tf.norm(est[:, 3:], ord=1),
                                                                            ord=2)

    def minimize(self, loss):

        return self.optimizer.minimize(loss)
