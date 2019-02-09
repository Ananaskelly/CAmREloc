import numpy as np

from utils.metrics import calc_pose_error, calc_qua_error
from utils.image_processing import inf_preprocess


class SimplePredictor:

    def __init__(self, model, test_batcher, sess):
        self.model = model
        self.test_batcher = test_batcher
        self.sess = sess

    def test(self):

        mean_pose_error = 0
        mean_qua_error = 0

        batch_size = self.test_batcher.b_size
        count = 0

        batch = self.test_batcher.next()
        while batch is not None:
            feed_dict = {
                self.model.x: batch[0]
            }
            loss, prediction = self.sess.run([self.model.loss, self.model.prediction], feed_dict=feed_dict)
            mean_pose_error += calc_pose_error(batch[1][:, :3], prediction[:, :3])
            mean_qua_error += calc_qua_error(batch[1][:, 3:], prediction[:, 3:])
            count += batch_size

            batch = self.test_batcher.next()

        mean_qua_error /= count
        mean_pose_error /= count

        return mean_pose_error, mean_qua_error

    def sample_predict(self, path_to_img):
        data = inf_preprocess(path_to_img)
        data = np.expand_dims(data, axis=0)

        feed_dict = {
            self.model.x: data
        }

        [pred] = self.sess.run([self.model.prediction], feed_dict=feed_dict)

        return pred[:3], pred[3:]
