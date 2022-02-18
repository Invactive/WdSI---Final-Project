import matplotlib.pyplot as plt
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import pandas


def load_data():
    # Get data to teach from .xml files
    path = os.getcwd()
    parent = os.path.dirname(path)
    ann_dir = os.path.join(parent, 'train/annotations')
    img_dir = os.path.join(parent, 'train', 'images')
    data = []
    for file in os.listdir(ann_dir):
        ann_dict = {}
        img_dict = {}
        xmin_tmp = []
        xmax_tmp = []
        ymin_tmp = []
        ymax_tmp = []
        root = ET.parse(os.path.join(ann_dir, file)).getroot()
        ann_dict['filename'] = file
        ann_dict['name'] = root.find('object/name').text
        for field in root.findall('object'):
            xmin_tmp.append(field.find('bndbox/xmin').text)
            xmax_tmp.append(field.find('bndbox/xmax').text)
            ymin_tmp.append(field.find('bndbox/ymin').text)
            ymax_tmp.append(field.find('bndbox/ymax').text)
            img_name = root.find('filename').text
        ann_dict['xmin'] = xmin_tmp
        ann_dict['xmax'] = xmax_tmp
        ann_dict['ymin'] = ymin_tmp
        ann_dict['ymax'] = ymax_tmp
        ann_dict['image'] = cv2.imread(os.path.join(img_dir, img_name))
        data.append(ann_dict)

    return data

def learn_bovw(data):
    dictionarySize = 128

    BOW = cv2.BOWKMeansTrainer(dictionarySize)

    sift = cv2.SIFT_create()
    # path to train/images
    for i in data:
        img = i['image']
        kp, dsc= sift.detectAndCompute(img, None)
        BOW.add(dsc)
        print(i)
    #dictionary created
    vocab = BOW.cluster()
    np.save('voc.npy', vocab)



def main():
    df = load_data()
    print(pandas.DataFrame(df))
    learn_bovw(df)









if __name__ == '__main__':
    main()












