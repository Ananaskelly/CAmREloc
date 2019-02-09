import argparse
import tensorflow as tf


from core.pose_net_model import PoseNetModel
from core.simple_trainer import SimpleTrainer
from core.batcher import Batcher


def run():

    train_batcher = Batcher(path_to_data='./data/images', path_to_csv='./data/train.csv', batch_size=4)
    valid_batcher = Batcher(path_to_data='./data/images', path_to_csv='./data/valid.csv', batch_size=4)
    pn_model = PoseNetModel()
    pn_model.build_model()

    sess = tf.Session()

    trainer = SimpleTrainer(pn_model, train_batcher, valid_batcher, sess, 10, 1)
    trainer.train()


def parse_args():
    pass


if __name__ == '__main__':
    run()
