import os
from lxml import etree
import numpy as np


def VOC2YOLO(class_num, voc_img_path, voc_xml_path, yolo_txt_save_path, yolo_img_save_path=None):
    xmls = os.listdir(voc_xml_path)
    xmls = [x for x in xmls if x.endswith('.xml')]
    if yolo_img_save_path is not None:
        if not os.path.exists(yolo_img_save_path):
            os.mkdir(yolo_img_save_path)
    if not os.path.exists(yolo_txt_save_path):
        os.mkdir(yolo_txt_save_path)

    for idx, one_xml in enumerate(xmls):
        print(f"Processing file: {one_xml}")
        xl = etree.parse(os.path.join(voc_xml_path, one_xml))
        root = xl.getroot()
        objects = root.findall('object')
        img_size = root.find('size')
        img_w, img_h = 0, 0

        if img_size:
            img_width = img_size.find('width')
            if img_width is not None:
                img_w = int(img_width.text)
            img_height = img_size.find('height')
            if img_height is not None:
                img_h = int(img_height.text)

        # Add image info to YOLO
        yolo_data = []

        for ob in objects:
            label = ob.find('name').text
            class_id = class_num.get(label, -1)
            if class_id == -1:
                print(f"Warning: Class '{label}' not found in class_num, skipping.")
                continue

            bbox = ob.find('bndbox')
            if bbox is None:
                print(f"Warning: No 'bndbox' found in {one_xml}, skipping this object.")
                continue

            # 获取四个顶点
            x0 = float(bbox.find('x0').text)
            y0 = float(bbox.find('y0').text)
            x1 = float(bbox.find('x1').text)
            y1 = float(bbox.find('y1').text)
            x2 = float(bbox.find('x2').text)
            y2 = float(bbox.find('y2').text)
            x3 = float(bbox.find('x3').text)
            y3 = float(bbox.find('y3').text)

            # 计算最小外接矩形
            vertices = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
            x_min = np.min(vertices[:, 0])
            x_max = np.max(vertices[:, 0])
            y_min = np.min(vertices[:, 1])
            y_max = np.max(vertices[:, 1])

            # 计算中心点、宽度和高度
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # 归一化
            x_center /= img_w
            y_center /= img_h
            width /= img_w
            height /= img_h

            # 添加到YOLO格式数据
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存YOLO格式的文本文件
        with open(os.path.join(yolo_txt_save_path, one_xml.replace(".xml", ".txt")), 'w') as f:
            f.write("\n".join(yolo_data))


if __name__ == '__main__':
    VOC2YOLO(
        class_num={
            'car': 0,
            'truck': 1,
            'traffic-sign': 2,
            'people': 3,
            'motor': 4,
            'bicycle': 5,
            'traffic-light': 6,
            'tricycle': 7,
            'bridge': 8,
            'bus': 9,
            'boat': 10,
            'ship': 11,
            'ignored': 12
        },  # 标签种类
        voc_img_path=r'E:\1-Data\DataSet\CODrone\test\images',  # 数据集图片文件夹存储路径
        voc_xml_path=r'E:\1-Data\DataSet\CODrone\test\labels',  # 标签xml文件夹存储路径
        yolo_txt_save_path=r'E:\1-Data\DataSet\CODrone\test\text'  # 将要生成的txt文件夹存储路径
    )