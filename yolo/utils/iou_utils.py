# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

import tensorflow as tf
import math


def tf_iou(tensor1, tensor2, mode="iou"):
    """
    计算 iou
    :param tensor1: shape=[N, 4] 4: x, y, w, h
    :param tensor2: shape=[N, 4] 4: x, y, w, h
    :return: iou
    """
    tensor1 = tf.cast(tensor1, tf.float32)
    tensor2 = tf.cast(tensor2, tf.float32)

    tensor1_xy = tensor1[..., 0:2]
    tensor1_wh = tensor1[..., 2:4]
    tensor1_x1y1 = tensor1_xy - tensor1_wh * 0.5
    tensor1_x2y2 = tensor1_xy + tensor1_wh * 0.5

    tensor2_xy = tensor2[..., 0:2]
    tensor2_wh = tensor2[..., 2:4]
    tensor2_x1y1 = tensor2_xy - tensor2_wh * 0.5
    tensor2_x2y2 = tensor2_xy + tensor2_wh * 0.5

    intersect_mins = tf.maximum(tensor1_x1y1, tensor2_x1y1)
    intersect_maxes = tf.minimum(tensor1_x2y2, tensor2_x2y2)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    tensor1_area = tensor1_wh[..., 0] * tensor1_wh[..., 1]
    tensor2_area = tensor2_wh[..., 0] * tensor2_wh[..., 1]
    union_area = tensor1_area + tensor2_area - intersect_area
    iou = intersect_area / union_area
    if mode == "iou":
        return iou

    enclose_x1y1 = tf.minimum(tensor1_x1y1, tensor2_x1y1)
    enclose_x2y2 = tf.maximum(tensor1_x2y2, tensor2_x2y2)
    enclose_wh = tf.maximum(enclose_x2y2 - enclose_x1y1, 0.)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / enclose_area
    if mode == "giou":
        return giou

    dd = tf.reduce_sum(tf.square(tensor1_xy - tensor2_xy), axis=-1)
    cc = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    diou = iou - dd / cc
    if mode == "diou":
        return diou

    v = tf.constant(4.0 / math.pi ** 2) * tf.square(tf.math.atan(tensor1_wh[..., 0] / tensor1_wh[..., 1]) \
                                                    - tf.math.atan(tensor2_wh[..., 0] / tensor2_wh[..., 1]))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    if mode == "ciou":
        return ciou


if __name__ == "__main__":
    a = [[[5, 5, 9, 10], [4, 4, 8, 8]], [[4, 4, 8, 8], [4, 4, 8, 8]]]
    b = [[6, 6, 8, 8], [5, 5, 9, 9]]

    a = tf.constant(a)
    b = tf.constant(b)

    print(a, b)
    iou = tf_iou(a, b, mode="iou")
    print(iou)

    giou = tf_iou(a, b, mode="giou")
    print(giou)

    diou = tf_iou(a, b, mode="diou")
    print(diou)

    ciou = tf_iou(a, b, mode="ciou")
    print(ciou)