# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import random
import cv2


def data_gen(annotation_lines, batch_size, input_shape, anchors, out_layers, num_classes):
    """
    数据生成器
    :param annotation_lines: 所有的标注信息
    :param batch_size: 数据生成的批次大小
    :param input_shape: 输入的图像尺寸
    :param anchors: 锚点框大小
    :param num_classes: 类别书
    :return: 返回一个批次的数据， 和网络的输出对应
    """
    # 一共多少数据
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, out_layers, num_classes)

        # name = random.randint(0,1e4)
        # image_data = (image_data * 255).astype(np.uint8)[0]
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("./imgs/%d.png"%name, image_data)
        yield (image_data, y_true)


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def addsalt_pepper(img, SNR):
    img_ = img.copy()
    h, w, d = img_.shape
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, d, axis=-1)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声
    return img_


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20):
    """
    处理图片和label信息， 并进行数据增强
    :param annotation_line:
    :param input_shape:
    :param random:
    :param max_boxes:
    :return: image_data: (416, 416, 3), box_data: (x1,y1,x2,y2)
    """
    # 读取一行数据， 并切分
    line = annotation_line.split()
    # 读取图片
    image = Image.open(line[0])
    # 图片的原始宽高
    iw, ih = image.size
    # 网络的输入宽高
    h, w = input_shape
    # 获取bboxes
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # 获取缩放比例
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        # 偏移坐标， 即原始图片的起始坐标（0,0）在输入尺寸上的位置
        dx = (w-nw)//2
        dy = (h-nh)//2

        # 图片等比例缩放
        image = image.resize((nw, nh), Image.BICUBIC)
        # 背景图
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # 图片贴到背景图上
        new_image.paste(image, (dx, dy))
        # 像素归一化
        image_data = np.array(new_image)/255.

        # 修正box的位置
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            # 框洗牌
            np.random.shuffle(box)
            # 每次20个框， 多余的舍去， 不够的补零
            if len(box) > max_boxes:
                box = box[:max_boxes]
            # 更新box坐标
            box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
            box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
            box_data[:len(box)] = box
        return image_data, box_data

    # resize image
    jitter = 0.1
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.9, 1.1)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = np.array(image)


    # 添加滤波和椒盐
    if rand() < .5:
        image = addsalt_pepper(image, rand(0.95, 1))
        r = rand()
        if r < .25:
            image = cv2.blur(image, (5, 5))
        elif r < .5:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        elif r < .75:
            image = cv2.medianBlur(image, 5)
        else:
            image = image
    else:
        r = rand()
        if r < .25:
            image = cv2.blur(image, (5, 5))
        elif r < .5:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        elif r < .75:
            image = cv2.medianBlur(image, 5)
        else:
            image = image
        image = addsalt_pepper(image, rand(0.95, 1))


    # distort image
    hue = .1
    sat = 1.5
    val = 1.5

    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, out_layers, num_classes):
    """
    :param true_boxes: 缩放到输入尺寸的label框
    :param input_shape: 输入图片的大小
    :param anchors: coco数据的anchors
    :param num_classes: 数据集的类别
    :return: yolo的输出
    """
    # 保证类别不会出错
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

    # 转成浮点型数据
    true_boxes = np.array(true_boxes, dtype='float32')
    # 转成int32数据
    input_shape = np.array(input_shape, dtype='int32')

    # 未进行归一化的中心点坐标， 以及wh
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # 把长宽以及中心点坐标进行归一化
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m = batchsize
    m = true_boxes.shape[0]

    # grid_shapes =  [[10, 20], [20, 40]]
    grid_shapes = [input_shape//{0: 32, 1: 16, 2: 8}[l] for l in range(out_layers)]

    # [(m, 10, 20, 3, 85) (m, 20, 40, 3, 85)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors)//out_layers, 5+num_classes), dtype='float32') for l in range(out_layers)]

    # 增加一维， shape：(6, 2) ---> (1, 6, 2)
    anchors = np.expand_dims(anchors, 0)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max

    # 查找边长大于0的
    valid_mask = (boxes_wh[..., 0] > 0) & (boxes_wh[..., 1] > 0)

    for b in range(m):
        # 清理全0行
        wh = boxes_wh[b, valid_mask[b]]
        # 增加一维
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        # 相交最小的坐标点, 最大的坐标点
        intersects_min = np.maximum(boxes_min, anchors_min)
        intersects_max = np.minimum(boxes_max, anchors_max)
        # 相交的宽高, 面积
        intersects_wh = np.maximum(intersects_max - intersects_min, 0.)
        intersects_area = intersects_wh[..., 0] * intersects_wh[..., 1]

        # box的面积
        box_area = wh[..., 0] * wh[..., 1]
        # anchor的面积
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # iou， 所有的框与anchors进行对比， 计算iou
        iou = intersects_area / (box_area + anchor_area - intersects_area)

        # 为每个真实的框 找到最佳的锚点,  有几个框， 找到几个anchor
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            # 第几个特征图上面
            l = out_layers - n // 3 - 1
            # 在特征图的第几层上面
            k = n % 3

            # grid_shapes =  [[13, 13], [26, 26], [52, 52]]
            # true_boxes.shape = (b, N, 5) 5: cx,cy,w,h,classes
            # 在哪个特征图上的哪个位置
            i = np.floor(true_boxes[b, t, 0]*grid_shapes[l][1]).astype('int32')
            j = np.floor(true_boxes[b, t, 1]*grid_shapes[l][0]).astype('int32')

            # 类别
            c = true_boxes[b, t, 4].astype('int32')

            # 更新坐标信息
            y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
            # 设置为正样本
            y_true[l][b, j, i, k, 4] = 1
            # 标记类别
            y_true[l][b, j, i, k, 5+c] = 1

    return y_true
