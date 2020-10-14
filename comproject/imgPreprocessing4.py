# -*- coding: utf-8 -*-
# @TIME     : 2020/10/12 11:43
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : imgPreprocessing4.py
# @Software : PyCharm

import cv2  #opencv 读取进来为BGR格式
import matplotlib.pyplot as plt
import numpy as np

# 定义图片显式的方法
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('bluecat.jpg',0)

kernel = np.ones((3,3),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 2)
dilation = cv2.dilate(img,kernel,iterations =1)
res = np.hstack((img,erosion))
res2 = np.hstack((img,dilation))

# cv_show('erosion',res)
# cv_show('dilation',res2)

mydog = cv2.imread('dog.jpg')
# cv_show('mydog',mydog)

kernel = np.ones((5,5),np.uint8)

erode_1 = cv2.erode(mydog,kernel,iterations = 1)
erode_2 = cv2.erode(mydog,kernel,iterations = 2)
erode_3 = cv2.erode(mydog,kernel,iterations = 3)
res3 = np.hstack((erode_1,erode_2,erode_3))
# cv_show('erode',res3)

dilate_1 = cv2.dilate(mydog,kernel,iterations = 1)
dilate_2 = cv2.dilate(mydog,kernel,iterations = 2)
dilate_3 = cv2.dilate(mydog,kernel,iterations = 3)
res4 = np.hstack((dilate_1,dilate_2,dilate_3))
# cv_show('dilate',res4)

# 开运算
opening = cv2.morphologyEx(mydog,cv2.MORPH_OPEN,kernel)
res5 = np.hstack((mydog,opening))
# cv_show('opening',res5)

# 闭运算
closing = cv2.morphologyEx(mydog,cv2.MORPH_CLOSE,kernel)
res6 = np.hstack((mydog,closing))
# cv_show('closing',res6)

# 梯度运算
dilated = cv2.dilate(img,kernel, iterations =1)
erosion = cv2.erode(img,kernel, iterations =1)
res7 = np.hstack((dilated, erosion, dilated-erosion))
# cv_show('Gradient', res7)

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
# cv_show('gradient',gradient)

# 礼帽与黑帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
res8 = np.hstack((img, tophat))
# cv_show('tophat', res8)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
res9 = np.hstack((img, blackhat))
# cv_show('blackhat', res9)

# 直方图
gcat = cv2.imread('bluecat.jpg',cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([gcat],[0],None,[256],[0,256])
# print(hist.shape)

hist = hist.flatten()
# print(hist.shape)

#print(np.arange(256))
#x = np.array([x for x in range(256)]);
#x = np.arange(256).reshape(256,1)
#x = np.transpose(x)
#print(x.shape)
#plt.plot(hist)

# plt.bar(np.arange(256),hist)

# plt.subplot(2,1,1),plt.imshow(gcat,'gray')
# plt.subplot(2,1,2),plt.hist(gcat.ravel(),256,[0,256]);

cat =  cv2.imread('bluecat.jpg')
color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([cat],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])

# mask操作
# mask = np.zeros(img.shape[:2], np.uint8)
#
# mask[100:300, 100:400] = 255
# masked_img = cv2.bitwise_and(img,img,mask = mask)
#
# # 计算掩码区域和非掩码区域的直方图
# # 检查作为掩码的第三个参数
# hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
# hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0,256])

# 直方图均衡化
# plt.hist(img.ravel(),256)
equ = cv2.equalizeHist(img)
# plt.hist(equ.ravel(),256)

# plt.show()

#均衡化对比
res10 = np.hstack((img,equ))
# cv_show('res',res10)

# 加上自适应均衡化的对比图
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
res11 = np.hstack((img,equ,cl1))
# cv_show('res11',res11)