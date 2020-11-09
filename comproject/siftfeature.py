# -*- coding: utf-8 -*-
# @TIME     : 2020/11/09 12:06
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : siftfeature.py
# @Software : PyCharm

import cv2  #opencv 读取进来为BGR格式
import matplotlib.pyplot as plt
import numpy as np

# print(cv2.__version__)
# 4.4.0

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('images/test2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv_show('gray',gray)

# 得到特征点
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img = cv2.drawKeypoints(gray,kp,img)
cv_show('keyPoint',img)

# 计算特征
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape)

print(des.shape)

print(des[0])