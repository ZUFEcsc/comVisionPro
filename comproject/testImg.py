# -*- coding: utf-8 -*-
# @TIME     : 2020/9/15 11:08
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : test.py
# @Software : PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 安装cv2第三方库测试
# img = cv2.imread(r"C:\Users\Administrator\Pictures\14.jpg")
# cv2.imshow("Image", img)
# cv2.waitKey (0)
# cv2.destroyAllWindows()
# print(cv2.__version__)


# 读取本地图片，并进行BGR到RGB的转换
# img = cv2.imread(r"C:\Users\Administrator\Pictures\temp.jpg")
# b,g,r = cv2.split(img)
# img2 = cv2.merge([r,g,b])

# 显示图片
# cv2.imshow("Image", img)
# cv2.waitKey (0)
# cv2.destroyAllWindows()

# 定义图片显式的方法
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv_show('cat',img)

# 查看图像大小
# print(img.shape)

# 将图像转换为灰度图像并查看图像大小
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)

# 显示灰度图像
# cv_show('cat',img_gray)

# 打印数字图像
# print(img_gray)

# 对比图像像素个数
# print('图像像素个数：'+str(img.size))
# print('灰色图像像素个数：'+str(img_gray.size))

# 保存图像到本地并显示
# cv2.imwrite('mycat.png',img)
# mycat = cv2.imread('mycat.png')
# plt.imshow(mycat)
# cv_show('mycat',mycat)

# 打印图像数据类型
# print(type(img_gray))
# print(img.dtype)


# 截取图像的部分并显示
img = cv2.imread('mycat.png')
# img2 = img[0:100,0:120]
# cv_show('original',img)
# cv_show('ROI',img2)

# 复制图像
cur_img = img.copy()
# 只保留R通道
# cur_img[:,:,0]=0
# cur_img[:,:,1]=0
# cv_show('R channel',cur_img)

# 只保留G通道
# cur_img[:,:,0]=0
# cur_img[:,:,2]=0
# cv_show('G channel',cur_img)

# 只保留B通道
cur_img[:,:,1]=0
cur_img[:,:,2]=0
cv_show('G channel',cur_img)

# 读取方式：灰度图像
# img_cat = cv2.imread('mycat.png',cv2.IMREAD_GRAYSCALE)
#
# top_size,down_size,left_size,right_size=(50,50,50,50)
#
# # plt.imshow(img_cat)
# # cv_show('cat',img_cat)
# replicate = cv2.copyMakeBorder(img_cat,top_size,down_size,left_size,right_size,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img_cat,top_size,down_size,left_size,right_size,cv2.BORDER_REFLECT)
# reflect_101 = cv2.copyMakeBorder(img_cat,top_size,down_size,left_size,right_size,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img_cat,top_size,down_size,left_size,right_size,cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img_cat,top_size,down_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)

# # 显示
# plt.subplot(2,3,1),plt.imshow(img_cat,'gray'),plt.title('original image')
# plt.subplot(2,3,2),plt.imshow(replicate,'gray'),plt.title('replicate')
# plt.subplot(2,3,3),plt.imshow(reflect,'gray'),plt.title('reflect')
# plt.subplot(2,3,4),plt.imshow(reflect_101,'gray'),plt.title('reflect_101')
# plt.subplot(2,3,5),plt.imshow(wrap,'gray'),plt.title('wrap')
# plt.subplot(2,3,6),plt.imshow(constant,'gray'),plt.title('constant')
# # plt.imshow()函数负责对图像进行处理，并显示其格式，
# # plt.show()则是将plt.imshow()处理后的函数显示出来。
# plt.show()

# # 数值计算
# img_cat = cv2.imread('mycat.png')
# img_dog = cv2.imread('mydog.jpg')

# print(img_cat[20:25,20:25,0])
# img_cat2 = img_cat+10
# print(img_cat2[20:25,20:25,0])
# print(img_dog[20:25,20:25,0])

# 重新设定图像大小
# img_cat = cv2.resize(img_cat,(200,200))  # h,w
# print(img_cat.shape)

# print((img_cat+img_dog)[0:5,0:5,0])
# print(cv2.add(img_cat,img_dog)[0:5,0:5,0])

# # 图像融合
# print(img_cat.shape)
# print(img_dog.shape)
# res = cv2.addWeighted(img_dog,0.1,img_cat,0.9,0)
# plt.imshow(res)
# plt.show()
#
# # 图像变形/图像缩放
# img_cat = cv2.resize(img_cat,(0,0),fx=1,fy=2)
# plt.imshow(img_cat)
# plt.show()