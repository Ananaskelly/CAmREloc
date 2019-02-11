import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

        mean_train_epoch_loss = 0
        train_losses = []
        valid_losses = []

        for i in range(num_it):
            loss, accuracy = self.train_step()
            mean_train_epoch_loss += loss

            if i % self.print_step == 0:
                print('Iteration: {}, loss: {}'.format(i, loss))

            if i % num_it_per_epoch == 0 and i != 0:
                v_loss, v_accuracy = self.valid_step()
                train_losses.append(mean_train_epoch_loss / num_it_per_epoch)
                valid_losses.append(v_loss)
                print('Validation loss: {}'.format(v_loss))
                print('Mean epoch train loss: {}'.format(mean_train_epoch_loss / num_it_per_epoch))
                mean_train_epoch_loss = 0

        idx = np.arange(len(train_losses))
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.title('кажется, у меня проблемы')
        plt.plot(idx, train_losses, c='r')
        plt.plot(idx, valid_losses, c='g')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('./results/pose_net_exp.png')

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
