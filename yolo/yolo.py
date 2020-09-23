# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

import tensorflow as tf
from tensorflow.keras.layers import Input
from yolo.backbone.cspdarknet53 import cspdarknet53
from yolo.utils.iou_utils import tf_iou
from yolo.configs import YOLO_Config

class YOLO(object):
    cfg = YOLO_Config()

    def model(self, cfg):
        model = cspdarknet53(Input(cfg.input_shape), cfg.num_anchors, cfg.num_classes)
        model.summary()

    def get_grid(self, shape):
        grid_shape = tf.cast(shape, tf.float32)
        z = tf.zeros((grid_shape[0], grid_shape[1]))
        x = tf.range(grid_shape[1]) + z
        y = tf.expand_dims(tf.range(grid_shape[0]), axis=-1) + z
        grid = tf.cast(tf.expand_dims(tf.stack([x, y], axis=-1), axis=2), tf.float32)
        return grid, grid_shape

    def pred_box(self, y_pred, anchors, num_classes, calc_loss=False):
        # 获取网格grid：结构是(10, 20, 1, 2)，数值为0~12的全遍历二元组；
        grid, grid_shape = self.get_grid(tf.shape(y_pred)[1:3])

        # 处理预测数据
        pred_xy, pred_wh, pred_obj, pred_class_probs = tf.split(y_pred, (2, 2, 1, num_classes), axis=-1)

        # 预测的pred_box_xy， 不是框的中心点xy， 而是在每一个格点上的偏移量x,y, 所以不能超过1， 要采用sigmoid归一化

        _xy = (tf.sigmoid(pred_xy) + tf.cast(grid, tf.float32)) / tf.cast(grid_shape[::-1], tf.float32)
        # 采用exp处理宽高
        _wh = tf.exp(pred_wh) * anchors / tf.cast(self.input_shape[::-1], tf.float32)

        pred_obj = tf.sigmoid(pred_obj)
        pred_class_probs = tf.sigmoid(pred_class_probs)

        # 合并变量， box
        pred_xywh = tf.concat((_xy, _wh), axis=-1)

        if calc_loss:
            return pred_xywh

        y1x1 = (_xy - _wh / 2)[::-1]
        y2x2 = (_xy + _wh / 2)[::-1]
        y1x1y2x2 = tf.concat([y1x1, y2x2], axis=-1)
        pred = tf.concat([y1x1y2x2, pred_obj, pred_class_probs], axis=-1)
        return tf.reshape(pred, (tf.shape(pred)[0], -1, tf.shape(pred)[-1]))


    def calculation_loss(self, y_true, y_pred, loss_mode="ciou"):
        """
        loss 函数
        :param y_true: 真实标签， [b, 13, 13, 3, 85] or [b, 26, 26, 3, 85] or [b, 52, 52, 3, 85] , 255 = 3 * (5 + 80)
        :param y_pred: 预测结果， [b, 13, 13, 3, 85] or [b, 26, 26, 3, 85] or [b, 52, 52, 3, 85]
        :return: loss结果
        """
        # 0. 获取网格grid：结构是(13, 13, 1, 2)，数值为0~12的全遍历二元组；
        grid, grid_shape = self.get_grid(tf.shape(y_pred)[1:3])

        # 1. 处理预测数据
        pred_xywh = self.pred_box(
            y_pred, self.cfg.anchors, self.cfg.num_classes, calc_loss=True)

        # 3. 处理正确label
        true_xy, true_wh, true_obj, true_class_probs = tf.split(
            y_true, (2, 2, 1, self.cfg.num_classes), axis=-1)

        # 将xy转成偏移量
        true_xy = true_xy * grid_shape[::-1] - grid
        true_wh = tf.math.log(true_wh / tf.cast(self.cfg.anchors, tf.float32) * tf.cast(self.cfg.input_shape[::-1], tf.float32))
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        # 给box更高的权重
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        # 4. 计算所有的masks
        # 去掉一维, 和expand_dims相反
        obj_mask = tf.squeeze(true_obj, -1)
        # true_box， 正确的框， shape = [n, 4]， n为框的个数
        true_box = tf.boolean_mask(y_true[..., :4], tf.cast(obj_mask, tf.bool))

        # 求计算的到的iou最大的框
        best_iou = tf.reduce_max(tf_iou(pred_xywh, true_box, mode="iou"), axis=-1)
        # 最大的iou小于阈值为副样本
        ignore_mask = tf.cast(best_iou < self.cfg.ignore_thresh, tf.float32)
        # 扩展维度， 方便计算
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # 5. 计算loss
        #xy_loss = box_loss_scale * true_obj * tf.nn.sigmoid_cross_entropy_with_logits(true_xy/1.1, y_pred[..., :2])
        #wh_loss = box_loss_scale * true_obj * tf.cast(0.5, tf.float32) * tf.square(true_wh - y_pred[..., 2:4])

        iou = tf.expand_dims(tf_iou(tf.concat([true_xy, true_wh], axis=-1), y_pred[..., 0:4], mode="ciou"), -1)
        xywh_loss = box_loss_scale * true_obj * (1 - iou)

        obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(true_obj, y_pred[..., 4:5])
        obj_loss = true_obj * obj_loss + (1 - true_obj) * obj_loss * ignore_mask

        class_loss = true_obj * tf.nn.sigmoid_cross_entropy_with_logits(
                     true_class_probs, y_pred[..., 5:])

        # 6. 求和： (batch, grid_x, grid_y, anchors, :) => (batch, 1)
        #xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3, 4))
        #wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3, 4))

        xywh_loss = tf.reduce_sum(xywh_loss, axis=(1, 2, 3, 4))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))

        return tf.cast(xywh_loss + obj_loss + class_loss, tf.float32)