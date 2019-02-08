import cv2
import numpy as np


def preprocess(path, shape, crop_type='center_crop'):

    if crop_type not in ['center_crop', 'random_crop']:
        raise Exception('Such crop_type is not supported!')

    img = cv2.imread(path)
    assert img.shape == shape, 'Shapes don\'t match: {} with {} for file {}'.format(img.shape, shape, path)
    img = cv2.resize(img, (320, 240))

    if crop_type == 'center_crop':
        return center_crop(img)
    elif crop_type == 'random_crop':
        return random_crop(img)


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
