import os
import numpy as np


def split(path_to_input_csv, out_path):
    """

    :param path_to_input_csv: full path to input *.csv file
    :param out_path:          root path to store output files
    :return:
    """

    with open(path_to_input_csv, 'r') as file:
        lines = file.readlines()

        n_lines = len(lines)
        n_train = int(n_lines*0.8)
        n_valid = int(n_lines*0.1)

        np.random.shuffle(lines)

        train_proto_path = os.path.join(out_path, 'train.csv')
        valid_proto_path = os.path.join(out_path, 'valid.csv')
        test_proto_path = os.path.join(out_path, 'test.csv')

        with open(train_proto_path, 'w') as train_file:
            train_file.writelines(lines[:n_train])
        with open(valid_proto_path, 'w') as valid_file:
            valid_file.writelines(lines[n_train:n_train + n_valid])
        with open(test_proto_path, 'w') as test_file:
            test_file.writelines(lines[n_train + n_valid:])


if __name__ == '__main__':
    split('../data/info.csv', '../data/')
