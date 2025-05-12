import os
import json
from PIL import Image

"""
首先满足YOLO格式要求：
        images:
            train
            val
            test
        labels:
            train
            val
            test
"""

# 保存位置
output_dir = r"E:\1-Data\DataSet\CODrone\coco"  # 修改为YOLO格式的数据集路径；
# 数据集路径
dataset_path = r"E:\1-Data\DataSet\CODrone"  # 修改你想输出的coco格式数据集路径
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# 类别映射
categories = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "truck"},
    {"id": 3, "name": "bus"},
    {"id": 4, "name": "traffic-sign"},
    {"id": 5, "name": "people"},
    {"id": 6, "name": "motor"},
    {"id": 7, "name": "bicycle"},
    {"id": 8, "name": "traffic-light"},
    {"id": 9, "name": "tricycle"},
    {"id": 10, "name": "bridge"},
    {"id": 11, "name": "boat"},
    {"id": 12, "name": "ship"}
]


# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]


# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }


# 处理每个数据集分区
# 只有train文件夹
# for split in ['train']:
# 只有train val文件夹
# for split in ['train',  'val',]:
# 只有train val test 文件夹
# for split in ['train',  'val', 'test']

for split in ['train',  'val', 'test']:
    coco_format = init_coco_format()
    annotation_id = 0

    for img_name in os.listdir(os.path.join(images_path, split)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_path, split, img_name)
            label_path = os.path.join(labels_path, split, img_name.replace("jpg", "txt"))

            img = Image.open(img_path)
            img_width, img_height = img.size
            image_info = {
                "file_name": img_name,
                "id": len(coco_format["images"]) + 0,
                "width": img_width,
                "height": img_height
            }
            coco_format["images"].append(image_info)

            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        category_id, x_center, y_center, width, height = map(float, line.split())
                        bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            # 根据你的数据集修改category_id是否需要减1或者加1
                            "category_id": int(category_id)+1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation)
                        annotation_id += 1
        # 每处理1000个图片时打印一次"正在处理"
        if (len(coco_format["images"]) + 1) % 1000 == 0:
            print("正在处理")

    # 为每个分区保存JSON文件
    with open(os.path.join(output_dir, f"{split}.json"), "w") as json_file:
        json.dump(coco_format, json_file, indent=4)
