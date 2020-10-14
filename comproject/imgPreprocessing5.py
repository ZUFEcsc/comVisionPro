# -*- coding: utf-8 -*-
# @TIME     : 2020/10/12 15:49
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : imgPreprocessing5.py
# @Software : PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('bluecat.jpg',0)

img_float32 = np.float32(img)
dft = cv2.dft(img_float32,flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

rows, cols = img.shape
crow, ccol = np.int32(rows/2) , np.int32(cols/2)

# 首先创建一个掩码，中心正方形为1，其余全为零(低通滤波器)
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# 应用掩码和逆DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)

img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# 图像梯度与边缘检测
cat = cv2.imread('bluecat.jpg',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(cat,cv2.CV_64F,1,0,ksize=3)
sobely= cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

# plt.imshow(sobelx,cmap = 'gray')
# plt.imshow(sobely,cmap = 'gray')
# plt.show()

sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# cv_show('sobelxy',sobelxy)

sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
# cv_show('sobelxy2',sobelxy)

# 不同算子的差异
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

laplacian  = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
# cv_show('res',res)

# canny算子
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)
res2 = np.hstack((v1,v2))
cv_show('res',res2)