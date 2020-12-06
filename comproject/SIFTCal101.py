# -*- coding: utf-8 -*-
# @TIME     : 2020/12/06 11:10
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : SIFTCal101.py
# @Software : PyCharm

'''
CV_INTER_NN - 最近邻插值,
CV_INTER_LINEAR - 双线性插值 (缺省使用)
CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..
CV_INTER_CUBIC - 立方插值
'''

import os, codecs
import numpy as np
from sklearn.cluster import KMeans
import cv2  #opencv 读取进来为BGR格式
import matplotlib.pyplot as plt

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_file_name():
    path_filenames = []
    filename_list = []

    root = os.getcwd()  # 获取当前路径
    data = '101_ObjectCategories'  # 101数据集的文件夹名称
    path = os.listdir(root + '/' + data)  # 显示该路径下所有文件
    path.sort()
    vp = 0.1  # 测试集合取总数据前10%
    ftr = open('train.txt', 'w')
    fva = open('val.txt', 'w')
    i = 0

    for line in path:
        subdir = root + '/' + data + '/' + line
        childpath = os.listdir(subdir)
        mid = int(vp * len(childpath))
        for child in childpath[:30]:
            fpath = data + '/' + line + '/'
            subpath = data + '/' + line + '/' + child;
            d = ' %s' % (i)
            t = subpath + d
            path_filenames.append(os.path.join(fpath, child))
            filename_list.append(child)
        i = i + 1
    return path_filenames


def knn_detect(file_list, cluster_nums, randomState=None):
    features = []
    files = file_list
    sift = cv2.SIFT()
    for file in files:
        print(file)
        img = cv2.imread(file)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.dtype)
        _, des = sift.detectAndCompute(gray, None)

        if des is None:
            file_list.remove(file)
            continue

        reshape_feature = des.reshape(-1, 1)
        features.append(reshape_feature[0].tolist())

    input_x = np.array(features)

    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_


def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in filenames]

    return dict(zip(files, labels))


def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))


path_filenames = get_file_name()

labels, cluster_centers = knn_detect(path_filenames, 101)

res_dict = res_fit(path_filenames, labels)
save('./', 'knn_res.txt', res_dict)
