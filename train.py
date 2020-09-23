from yolo.yolo import YOLO
from yolo.configs import Config


cfg = Config()
yolo = YOLO(
    input_shape=(320, 320, 3),
    anchors=cfg.anchors,
    anchors_mask=cfg.anchors_mask,
    num_classes=cfg.num_classes,

)

model = yolo.model()
