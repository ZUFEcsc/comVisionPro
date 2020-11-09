# -*- coding: utf-8 -*-
# @TIME     : 2020/10/26 11:35
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : imgPreprocessing6.py
# @Software : PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('test2.jpg')

# cv_show('lena',img)
# print(img.shape)
# print(img)

up = cv2.pyrUp(img)
# cv_show('up',up)
# print(up.shape)

down = cv2.pyrDown(img)
# cv_show('down',down)
# print(down.shape)

up2 = cv2.pyrUp(up)
# cv_show('up2',up2)
# print(up2.shape)

up_down = cv2.pyrDown(up)
# cv_show('img_up_down',up_down)
# print(up_down.shape)
# print(up_down)

res = np.hstack((img,up_down))
# cv_show('res',res)
# cv_show('img-updown',img-up_down)

res2 = img - up_down
# cv_show('Laplace',res2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY)
# cv_show('thresh', thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(np.array(contours).shape)

draw_img = img.copy()
res = cv2.drawContours(img,contours,-1,(0,0,225),2)
# cv_show('res',res)
# cv_show('img',img)

cnt = contours[0]
# print(cnt)

# 面积
area = cv2.contourArea(cnt)
# print(area)

# 轮廓周长
perimeter = cv2.arcLength(cnt,True)
# print(perimeter)

# 特征矩
M = cv2.moments(cnt)
# print( M )

# 质心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# print(cx,cy)

img2 = cv2.imread('images/star.png')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv_show('img_2',img2)

ret,thresh = cv2.threshold(img2_gray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

draw_img = img2.copy()
res = cv2.drawContours(img2,contours, -1, (0,0,255),2)
# cv_show('res', res)

cnt = contours[0]
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

draw_img = img2.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0,255,0),2)
# cv_show('res', res)

# 模板匹配
img = cv2.imread('images/lena.jpg',0)
img2 = img.copy()
template = cv2.imread('images/face.jpg',0)
h, w = template.shape[:2]
# print(img.shape)
# print(template.shape)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
# print(res.shape)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# print(min_val)
# print(min_loc)
# print(max_val)
# print(max_loc)

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # 应用模板匹配
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

img_rgb = cv2.imread('images/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('images/mario_coin.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

# 取匹配程度大于80%的坐标
threshold = 0.8
loc = np.where( res >= threshold)
img_rgb2 = img_rgb.copy()
for pt in zip(*loc[::-1]):  #*号表示可选参数
    cv2.rectangle(img_rgb2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imwrite('res.png',img_rgb)
# cv_show('img_rgb',img_rgb2)

#res = np.stack((img_rgb,img_rgb2))
# cv_show('res',res)