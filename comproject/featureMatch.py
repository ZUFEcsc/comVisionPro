# -*- coding: utf-8 -*-
# @TIME     : 2020/11/09 12:30
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : featureMatch.py
# @Software : PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img1 = cv2.imread('images/box.png',0)
img2 = cv2.imread('images/box_in_scene.png',0)

# cv_show('img1',img1)
# cv_show('img2',img2)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# print(des1.shape)
# print(des2.shape)

#crossCheck表示两个特征点要互相匹配，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是最近的。
# NORM_L2:归一化数组的欧式距离，入托企图特征计算方法需要考虑不同的距离

bf = cv2.BFMatcher(crossCheck=True)

# 1对1匹配
matches = bf.match(des1,des2)
matches = sorted(matches, key =lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
# cv_show('img3',img3)

# k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# 应用比例测试
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn将列表作为匹配项。
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# cv_show('img3',img3)
