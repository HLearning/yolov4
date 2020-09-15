# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

import json
import os
from collections import defaultdict


def get_box(annotations_path, images_path):
    name_box_id = defaultdict(list)
    f = open(annotations_path, encoding='utf-8')
    data = json.load(f)
    annotations = data['annotations']
    for ant in annotations:
        image_id = ant['image_id']
        name = os.path.join(images_path, '%012d.jpg' % image_id)
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])
    return name_box_id


def write_file(file_name, name_box_id):
    f = open(file_name, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])
            box_info = " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()


if __name__ == "__main__":
    annotations_path = "../../../datasets/coco2017/annotations/instances_val2017.json"
    images_path = "../../../datasets/coco2017/val2017"
    file_name = 'val.txt'
    name_box_id = get_box(annotations_path, images_path)
    write_file(file_name, name_box_id)