import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import os
import pandas


def load_data():
    # Get data to teach from .xml files
    path = os.getcwd()
    parent = os.path.dirname(path)
    # print("Parent directory", parent)
    ann_dir = os.path.join(parent, 'train/annotations')
    # print("Annot directory", ann_dir)
    ann_list = []
    # print(os.listdir(ann_dir))
    for file in os.listdir(ann_dir):
        ann_dict = {}
        xmin_tmp = []
        xmax_tmp = []
        ymin_tmp = []
        ymax_tmp = []
        root = ET.parse(os.path.join(ann_dir, file)).getroot()
        ann_dict['filename'] = file
        for field in root.findall('object'):
            xmin_tmp.append(field.find('bndbox/xmin').text)
            xmax_tmp.append(field.find('bndbox/xmax').text)
            ymin_tmp.append(field.find('bndbox/ymin').text)
            ymax_tmp.append(field.find('bndbox/ymax').text)
        ann_dict['xmin'] = xmin_tmp
        ann_dict['xmax'] = xmax_tmp
        ann_dict['ymin'] = ymin_tmp
        ann_dict['ymax'] = ymax_tmp
        ann_list.append(ann_dict)
    return pandas.DataFrame(ann_list)




def main():
    df = load_data()
    print(df)











if __name__ == '__main__':
    main()












