# -*- coding: utf-8 -*-
# @TIME     : 2020/9/28 15:15
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : imgPreprocessing.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img_np = np.fromfile('mycat.png',dtype = np.uint8)
# img = cv2.imdecode(img_np,-1)

img = cv2.imread('mycat.png')
cv_show('原图',img)

# OpenCV读取的图片从BGR转换为RGB
b, g, r =cv2.split(img)
img2 = cv2.merge([r,g,b])
cv_show('RGB',img2)

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
cv_show('缩放',res)

# print(img.shape)
rows,cols,rgb = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv_show('平移',dst)

rows,cols,rgb = img.shape
# # cols-1 和 rows-1 是坐标限制
M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),45,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv_show('旋转',dst)

# 仿射变换
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

# 透视变换
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()