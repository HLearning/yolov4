# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================


class Config(object):
    batch = 64
    # 输入图片尺寸
    width = 608
    height = 608
    channels = 3
    learning_rate = 0.0013

    # 锚点
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
    # 锚点的个数
    num_anchors = 9
    anchors_mask = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # coco数据集的类别数
    num_classes = 80

    # 副样本产生的阈值
    ignore_thresh = 0.7
    # loss函数
    iou_loss = 'ciou'
    beta_nms = 0.6