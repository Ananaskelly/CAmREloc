import numpy as np


from utils.image_processing import preprocess


class Batcher:

    def __init__(self, path_to_data, path_to_csv, batch_size):
        self.data_path = path_to_data
        self.csv_path = path_to_csv
        self.batch_size = batch_size

        self.data = []
        self.num_ex = 0
        self.load_set()

        self.global_idx = 0
        self.generator = self.__iter__()

    def load_set(self):

        with open(self.csv_path, 'r') as csv_file:
            lines = csv_file.readlines()
            self.num_ex = len(lines)

            for line in lines:
                info = line.split(',')
                img_path = info[-3]
                px, py, pz = info[3:6]
                qw, qx, qy, qz = info[6:10]
                sample_dict = {
                    'pos': [px, py, pz],
                    'qua': [qw, qx, qy, qz],
                    'img': preprocess(img_path, (480, 640), 'center_crop')
                }

                self.data.append(sample_dict)

    def next_batch(self):

        return next(self.generator)

    def __iter__(self):
        while True:
            if self.global_idx + self.batch_size < self.num_ex:
                self.global_idx = 0
                np.random.shuffle(self.data)

            idx = self.global_idx
            self.global_idx += self.batch_size
            yield self.data[idx:idx + self.batch_size]
