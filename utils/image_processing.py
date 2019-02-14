import cv2
import os
import numpy as np


def inf_preprocess(path):
    img = cv2.imread(path)

    if img is None:
        raise Exception('Can\'t load file {}'.format(path))

    img = cv2.resize(img, (320, 240))

    return center_crop(img)


def preprocess(path, shape, crop_type='center_crop', norm_type=None):

    if crop_type not in ['center_crop', 'random_crop']:
        raise Exception('Such crop_type is not supported!')

    img = cv2.imread(path)

    if img is None:
        raise Exception('Something went wrong with file {}'.format(path))

    if img.shape != shape:
        raise Exception('Shapes don\'t match: {} with {} for file {}'.format(img.shape, shape, path))

    if norm_type is not None:
        if norm_type == 'range':
            img = range_normalization(img, -1, 1)
        elif norm_type == 'mean_std':
            img = mean_std_normalization(img)
        else:
            print('Such normalization type is not supported; {}'.format(norm_type))

    img = cv2.resize(img, (320, 240))

    if crop_type == 'center_crop':
        return center_crop(img)
    elif crop_type == 'random_crop':
        return random_crop(img)


def mean_std_normalization(img):

    mean = np.load('../ex_data/mean.npy')
    std = np.load('../ex_data/std.npy')

    return (img - mean)/std


def range_normalization(img, x, y):

    _, _, ch = img.shape
    ran = y - x
    for i in range(ch):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i]))/(np.max(img[:, :, i]) - np.min(img[:, :, i]))
        img[:, :, i] *= ran
        img[:, :, i] += x

    return img


def random_crop(img, base_size=224):
    img_height, img_width, _ = img.shape
    h_margin = img_height - base_size
    w_margin = img_width - base_size

    h_left_offset = np.random.randint(0, h_margin)
    w_left_offset = np.random.randint(0, w_margin)

    return img[h_left_offset:h_left_offset+base_size, w_left_offset:w_left_offset+base_size]


def center_crop(img, base_size=224):
    img_height, img_width, _ = img.shape

    h_margin = img_height - base_size
    w_margin = img_width - base_size

    h_left_offset = h_margin - h_margin // 2
    w_left_offset = w_margin - w_margin // 2

    return img[h_left_offset:h_left_offset+base_size, w_left_offset:w_left_offset+base_size]


def get_base_mean_std(files, data_path, save_path):

    data = []
    for file in files:
        with open(file, 'r') as csv_file:
            lines = csv_file.readlines()
            for line in lines:
                data.append(cv2.imread(os.path.join(data_path, line.split(',')[-3])))

    mean = np.mean(np.stack(data), axis=0)
    std = np.std(np.stack(data), axis=0)

    print('mean_shape: {},\n std_shape: {}'.format(mean.shape, std.shape))

    np.save(os.path.join(save_path, 'mean_img'), mean)
    np.save(os.path.join(save_path, 'std_img'), std)


'''
def test(path):
    img = cv2.imread(path)
    img2 = cv2.resize(img, (320, 240))
    print(img2.shape)
    img2 = img2[8:8+224, 48:48+224]
    cv2.imshow('test pic', img2)
    cv2.waitKey(0)
    print(img.shape)


if __name__ == '__main__':
    test('../data/images/img_0_0_1542099066033315900.png')
'''

if __name__ == '__main__':
    get_base_mean_std(['../data/train.csv', '../data/valid.csv'], '../data/images', '../ex_data')
