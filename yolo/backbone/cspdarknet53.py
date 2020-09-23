# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, LeakyReLU, BatchNormalization, Lambda, Input
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.activations import linear
from tensorflow.keras.models import Model
from yolo.activation import Mish


def Conv2D_BN_Act(x, filters, size, strides=1, batch_norm=True, activation="mish"):
    padding = ('same' if strides == 1 else 'valid')
    x = Conv2D(filters=filters,
               kernel_size=size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=l2(5e-4)
               )(x)
    if batch_norm:
        x = BatchNormalization()(x)

    if activation in ["Mish", "mish"]:
        x = Mish()(x)
    elif activation in ["LeakyReLU", "leakyrelu"]:
        x = LeakyReLU(alpha=0.1)(x)
    elif activation in ["Linear", "linear"]:
        x = linear(x)
    return x


def res_block(x, filters1, filters2, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D_BN_Act(x, filters1, (3, 3), strides=2, activation="mish")
    x_short = Conv2D_BN_Act(x, filters2, (1, 1))
    x_main = Conv2D_BN_Act(x, filters2, (1, 1))

    for i in range(num_blocks):
        y = Conv2D_BN_Act(x_main, filters1 // 2, (1, 1), activation="mish")
        y = Conv2D_BN_Act(y, filters2, (3, 3), activation="mish")
        x_main = Add()([x_main, y])

    x_main = Conv2D_BN_Act(x_main, filters2, (1, 1), activation="mish")
    x = Concatenate()([x_short, x_main])
    x = Conv2D_BN_Act(x, filters1, (1, 1), activation="mish")
    return x


def conv_block(x, filters):
    x = Conv2D_BN_Act(x, filters, (1, 1), activation="leakyrelu")
    x = Conv2D_BN_Act(x, filters * 2, (3, 3), activation="leakyrelu")
    x = Conv2D_BN_Act(x, filters, (1, 1), activation="leakyrelu")
    x = Conv2D_BN_Act(x, filters * 2, (3, 3), activation="leakyrelu")
    x = Conv2D_BN_Act(x, filters, (1, 1), activation="leakyrelu")
    return x


def spp(x):
    x1 = MaxPooling2D(strides=1, pool_size=5, padding='same')(x)
    x2 = MaxPooling2D(strides=1, pool_size=9, padding='same')(x)
    x3 = MaxPooling2D(strides=1, pool_size=13, padding='same')(x)
    x = Concatenate()([x1, x2, x3, x])
    return x


def darknet_backbone(x):
    x = Conv2D_BN_Act(x, 32, (3, 3), activation="mish")
    x = res_block(x, 64, 64, 1)
    x = res_block(x, 128, 64, 2)
    x = res_block(x, 256, 128, 8)
    x = res_block(x, 512, 256, 8)
    x = res_block(x, 1024, 512, 4)
    return x


def neck_layers(x):
    x = Conv2D_BN_Act(x, 512, (1, 1), activation='leakyrelu')
    x = Conv2D_BN_Act(x, 1024, (3, 3), activation='leakyrelu')
    x = Conv2D_BN_Act(x, 512, (1, 1), activation='leakyrelu')
    x = spp(x)
    x = Conv2D_BN_Act(x, 512, (1, 1), activation='leakyrelu')
    x = Conv2D_BN_Act(x, 1024, (3, 3), activation='leakyrelu')
    x = Conv2D_BN_Act(x, 512, (1, 1), activation='leakyrelu')
    return x


def cspdarknet53(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_backbone(inputs))
    x = darknet.get_layer('mish_71').output
    x = neck_layers(x)

    y1 = Conv2D_BN_Act(x, 256, (1, 1), activation='leakyrelu')
    y1_upsample = UpSampling2D(2)(y1)

    y2 = darknet.get_layer('mish_58').output
    y2 = Conv2D_BN_Act(y2, 256, (1, 1), activation='leakyrelu')
    y2 = Concatenate()([y2, y1_upsample])
    y2 = conv_block(y2, 256)
    y2_upsample = UpSampling2D(2)(Conv2D_BN_Act(y2, 128, (1, 1), activation='leakyrelu'))

    y3 = darknet.get_layer('mish_37').output
    y3 = Conv2D_BN_Act(y3, 128, (1, 1), activation='leakyrelu')
    y3 = Concatenate()([y3, y2_upsample])
    y3 = conv_block(y3, 128)

    y3_output = Conv2D_BN_Act(y3, 256, (3, 3), activation='leakyrelu')
    y3_output = Conv2D_BN_Act(y3_output, num_anchors * (num_classes + 5), (1, 1), batch_norm=False, activation="linear")

    y3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(y3)
    y3_downsample = Conv2D_BN_Act(y3_downsample, 256, (3, 3), strides=(2, 2), activation='leakyrelu')
    y2 = Concatenate()([y3_downsample, y2])
    y2 = conv_block(y2, 256)

    y2_output = Conv2D_BN_Act(y2, 512, (3, 3), activation='leakyrelu')
    y2_output = Conv2D_BN_Act(y2_output, num_anchors * (num_classes + 5), (1, 1), batch_norm=False, activation="linear")

    y2_downsample = ZeroPadding2D(((1, 0), (1, 0)))(y2)
    y2_downsample = Conv2D_BN_Act(y2_downsample, 512, (3, 3), strides=(2, 2), activation='leakyrelu')
    y1 = Concatenate()([y2_downsample, x])
    y1 = conv_block(y1, 512)

    y1_output = Conv2D_BN_Act(y1, 1024, (3, 3), activation='leakyrelu')
    y1_output = Conv2D_BN_Act(y1_output, num_anchors * (num_classes + 5), (1, 1), batch_norm=False, activation="linear")

    model = Model(inputs, [y1_output, y2_output, y3_output])
    return model


if __name__ == '__main__':
    inputs = Input((608, 608, 3))
    num_anchors = 9
    num_classes = 80
    model = cspdarknet53(inputs, num_anchors, num_classes)
    model.summary()
