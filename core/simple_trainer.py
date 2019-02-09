import tensorflow as tf


class SimpleTrainer:

    def __init__(self, model, train_batcher, valid_batcher, sess, epoch_num, print_fr):
        self.model = model
        self.train_batcher = train_batcher
        self.valid_batcher = valid_batcher
        self.sess = sess
        self.epoch_num = epoch_num

        self.print_step = print_fr

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):

        all_ex = self.train_batcher.num_ex
        batch_size = self.train_batcher.b_size

        num_it = self.epoch_num * (all_ex // batch_size)
        num_it_per_epoch = (all_ex // batch_size)

        for i in range(num_it):
            loss, accuracy = self.train_step()

            if i % self.print_step == 0:
                print('Iteration: {}, loss: {}, accuracy: {}'.format(i, loss, accuracy))

            if i % num_it_per_epoch == 0:
                loss, accuracy = self.valid_step()
                print('Validation loss: {}, validation accuracy: {}'.format(loss, accuracy))

    def train_step(self):
        batch_x, batch_y = self.train_batcher.next_batch()
        feed_dict = {
            self.model.x: batch_x,
            self.model.y: batch_y,
        }
        loss, _ = self.sess.run([self.model.loss, self.model.optimize], feed_dict=feed_dict)

        return loss, 0

    def valid_step(self):
        batch_x, batch_y = self.valid_batcher.next_batch()
        feed_dict = {
            self.model.x: batch_x,
            self.model.y: batch_y,
        }
        loss, _ = self.sess.run([self.model.loss, self.model.optimize], feed_dict=feed_dict)

        return loss, 0
