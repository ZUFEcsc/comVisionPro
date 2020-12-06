# -*- coding: utf-8 -*-
# @TIME     : 2020/12/06 11:12
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : OsTrainTest.py
# @Software : PyCharm
import os

root = os.getcwd()  # 获取当前路径
data = '101_ObjectCategories'  # 101数据集的文件夹名称
path = os.listdir(root + '/' + data)  # 显示该路径下所有文件
path.sort()
ftr = open('train.txt', 'w')
fva = open('val.txt', 'w')
i = 0

for line in path:
    subdir = root + '/' + data + '/' + line
    childpath = os.listdir(subdir)
    for child in childpath[:30]:
        subpath = data + '/' + line + '/' + child;
        d = ' %s' % (i)
        t = subpath + d
        fva.write(t + '\n')
    for child in childpath[30:]:
        subpath = data + '/' + line + '/' + child;
        d = ' %s' % (i)
        t = subpath + d
        ftr.write(t + '\n')
    i = i + 1

ftr.close()  # 关闭文件流
fva.close()