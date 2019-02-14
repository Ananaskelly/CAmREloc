import argparse
import tensorflow as tf


from core.hourglass_model import HourglassModel
from core.simple_trainer import SimpleTrainer
from core.simple_predictor import SimplePredictor
from core.batcher import Batcher


def run():

    train_batcher = Batcher(path_to_data='./data/images', path_to_csv='./data/train.csv', batch_size=4)
    valid_batcher = Batcher(path_to_data='./data/images', path_to_csv='./data/valid.csv', batch_size=4)
    model = HourglassModel()
    model.build_model()

    sess = tf.Session()

    trainer = SimpleTrainer(model, train_batcher, valid_batcher, sess, 100, 10)
    trainer.train(enable_phase=True)

    test_batcher = Batcher(path_to_data='./data/images', path_to_csv='./data/test.csv', batch_size=4,
                           batcher_type='test', norm_type='range')
    predictor = SimplePredictor(model, test_batcher, sess)
    print('Start testing...')
    mean_pos, mean_qua = predictor.test(enable_phase=True)
    print('Mean pose error: {}, mean quaternion error: {}'.format(mean_pos, mean_qua))


def parse_args():
    pass


if __name__ == '__main__':
    run()
