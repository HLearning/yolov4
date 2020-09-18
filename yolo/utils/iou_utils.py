# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

import tensorflow as tf
import math


def box_iou(b1, b2):
    """
    计算box的iou
    :param b1: tensor, shape=(i1,...,iN, 4), xywh
    :param b2: tensor, shape=(j, 4), xywh
    :return:
    """
    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def tf_iou(tensor1, tensor2, mode="iou", epsilon=1e-7):
    """
    计算 iou
    :param tensor1: shape=[N, 4] 4: x, y, w, h
    :param tensor2: shape=[N, 4] 4: x, y, w, h
    :param mode: iou类型
    :param epsilon: 极小值， 防止除零错误
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
    iou = intersect_area / tf.maximum(union_area, epsilon)
    if mode.lower() == "iou":
        return iou

    enclose_x1y1 = tf.minimum(tensor1_x1y1, tensor2_x1y1)
    enclose_x2y2 = tf.maximum(tensor1_x2y2, tensor2_x2y2)
    enclose_wh = tf.maximum(enclose_x2y2 - enclose_x1y1, 0.)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / tf.maximum(enclose_area, epsilon)
    if mode.lower() == "giou":
        return giou

    dd = tf.reduce_sum(tf.square(tensor1_xy - tensor2_xy), axis=-1)
    cc = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    diou = iou - dd / tf.maximum(cc, epsilon)
    if mode.lower() == "diou":
        return diou

    tensor1_wh_scale = tensor1_wh[..., 0] / tf.maximum(tensor1_wh[..., 1], epsilon)
    tensor2_wh_scale = tensor2_wh[..., 0] / tf.maximum(tensor2_wh[..., 1], epsilon)
    v = 4.0 / math.pi ** 2 * tf.square(tf.math.atan(tensor1_wh_scale) - tf.math.atan(tensor2_wh_scale))
    alpha = tf.stop_gradient(v / tf.maximum(1 - iou + v, epsilon))
    ciou = diou - alpha * v
    if mode.lower() == "ciou":
        return ciou


if __name__ == "__main__":
    a = [[[5, 5, 9, 10], [4, 4, 8, 8]],
         [[4, 4, 8, 8], [4, 4, 8, 8]]]
    b = [[[6, 6, 8, 8], [5, 5, 9, 9]],
        [[6, 6, 8, 8], [5, 5, 9, 9]]]

    a = tf.constant(a)
    b = tf.constant(b)

    print(a, b)
    iou = tf_iou(a, b, mode="iou")
    print(iou)
    #
    giou = tf_iou(a, b, mode="giou")
    print(giou)
    #
    # diou = tf_iou(a, b, mode="diou")
    # print(diou)
    #
    # ciou = tf_iou(a, b, mode="ciou")
    # print(ciou)