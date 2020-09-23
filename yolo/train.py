"""
@Time    : 19-11-28 下午1:16
@Author  : huangjinlei
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.yolo import YOLO
from yolo3.data.data_gen import data_gen
from yolo3.data.data_gen_hjl import data_gen_val
import numpy as np


def main():
    data_path = "/home/hjl/data2/datasets/train_202009"
    # 类别
    categories = ['pig']
    # 类别数
    num_classes = len(categories)
    # 图片的输入尺寸, hw
    input_shape = (320, 320)
    # coco数据集在yolo上的anchors, wh
    anchors_mask = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    anchors = np.array([[24,  31.],
                        [43,  49],
                        [56,  74],
                        [75,  56],
                        [100,  70],
                        [141,  78],
                        [154,  58],
                        [184,  70],
                        [216,  85]])
    # 用了3个网络层
    out_layers = len(anchors_mask)

    # 训练数据的批次大小
    batch_size = 16
    ignore_thresh = 0.5
    learning_rate_base = 1e-4
    log_dir = 'logs/'

    # 创建模型
    model = YOLO(input_shape=input_shape,
                 anchors=anchors,
                 anchors_mask=anchors_mask,
                 num_classes=num_classes,
                 ignore_thresh=ignore_thresh,
                 freeze=False,
                 loss="ciou"
                 ).model('xception', training=True)
    model.summary()
    model.load_weights("/home/hjl/data2/skyeyed/yolo3/logs/diou_model-loss0.6251-val_loss0.6233.h5")

    # 生成日志信息 以及 配置回调函数
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "model_epoch:{epoch:03d}-loss:{loss:.4f}-val_loss:{val_loss:.4f}.h5",
                                 monitor='loss', mode='min', save_weights_only=False, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)

    # 定义loss
    loss = [YOLO(input_shape=input_shape,
                 anchors=anchors[mask],
                 anchors_mask=anchors_mask,
                 num_classes=num_classes,
                 ignore_thresh=ignore_thresh
                 ).calculation_loss for mask in anchors_mask]
    model.compile(optimizer=Adam(lr=learning_rate_base), loss=loss)

    val_annotation_path = '/home/hjl/data2/skyeyed/yolo3/data/val_30k.txt'
    # 读取数据
    #with open(val_annotation_path) as f:
    #    train_lines = f.readlines()


    model.fit_generator(data_gen(data_path, batch_size, input_shape, anchors, out_layers, num_classes),
                        steps_per_epoch=500,
                        validation_data=data_gen(data_path, batch_size, input_shape, anchors, out_layers, num_classes),
                        validation_steps=100,
                        epochs=100,
                        initial_epoch=10,
                        callbacks=[logging, checkpoint, early_stopping],
                        #use_multiprocessing=True,
                        #workers=8
                        )

    # model.fit_generator(data_gen_val(train_lines, batch_size, input_shape, anchors, out_layers, num_classes),
    #                     steps_per_epoch=len(train_lines) // batch_size,
    #                     validation_data=data_gen_val(train_lines, batch_size, input_shape, anchors, out_layers, num_classes),
    #                     validation_steps=10,
    #                     epochs=100,
    #                     initial_epoch=0,
    #                     callbacks=[logging, checkpoint, early_stopping],
    #                        use_multiprocessing = True,
    #                       workers = 8
    #                     )


    model.save(log_dir + 'trained_weights.h5')


if __name__ == '__main__':
    main()
