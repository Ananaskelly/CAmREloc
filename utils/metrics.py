import numpy as np


def calc_pose_error(p1, p2):
    return np.sqrt((p1 - p2) ** 2)


def calc_qua_error(q1, q2):
    q2 = q2 / np.linalg.norm(q2)
    """
    ?????
    http://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
    (https://math.stackexchange.com/questions/90081/quaternion-distance)
    :return:
    """
    return 2 * np.arccos(np.abs(np.sum(q1 * q2, axis=1)))
