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
    # print(os.listdir(anno_dir))
    for file in os.listdir(ann_dir):
        ann_dict = {}
        root = ET.parse(os.path.join(ann_dir, file)).getroot()
        ann_dict['filename'] = file
        ann_dict['xmin'] = root.find('object/bndbox/xmin').text
        ann_dict['xmax'] = root.find('object/bndbox/xmax').text
        ann_dict['ymin'] = root.find('object/bndbox/ymin').text
        ann_dict['ymax'] = root.find('object/bndbox/ymax').text
        # print(ann_dict)
        ann_list.append(ann_dict)
    return pandas.DataFrame(ann_list)




def main():
    df = load_data()
    print(df)










if __name__ == '__main__':
    main()












