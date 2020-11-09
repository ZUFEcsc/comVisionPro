# -*- coding: utf-8 -*-
# @TIME     : 2020/10/31 12:03
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : HOG.py
# @Software : PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import hog

def rgb2gray(im): #灰度化
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def HOG(img):
    img_gray = rgb2gray(img)/255.0 #转化成灰度图像并进行归一化
    fd = hog(img_gray, orientations=10, pixels_per_cell=[8,8], cells_per_block=[8,8], visualize=False, 
             transform_sqrt=True,block_norm='L2-Hys')
    #print(fd)
    return fd

def distance_HOG(img1,img2):
    return np.sqrt(np.sum(np.square(img1 - img2))) #欧式距离

def drawline(recall,precision):
    plt.plot(recall,precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title('PR Graph of HOG')
    plt.show()

if __name__ == "__main__":
    img_index = 0 #设置样本img的编号
    img=cv2.imread('Corel-1K/image.orig/'+str(img_index)+'.jpg') #这里就是读取0.jpg
    img = cv2.resize(img,(200,200))
    img_class = int((img_index)/100)+1 #算出样本img的类别，0-99为类1,100-199为类2,，以此类推
    print("class",img_class)
    print('img',img.shape)
    
    HOG_img1 = HOG(img) #对img进行HOG处理
    print("HOG_img1 size",HOG_img1.shape)
    hog_index=[] 
    hog_img=[]
    hog_class = []
    hog_new_class = [] 
    
    for i in range(0,1000):
        if i == img_index: #如果读到样本img的编号就跳过
            print("skip>>>>>>>"+str(i))
            continue
        img2 = cv2.imread('.\\Corel-1K\\image.orig\\%d.jpg' %(i))
        img2 = cv2.resize(img2,(200,200))
        #print(img2.shape)
        #print(i)
        img2_class = int((i)/100)+1
        #print(img2_class)
        if img_class == img2_class: #判断循环取出的图片是否和样本为同一个类别
            new_class = 1
        else:
            new_class = 0
        HOG_img2 = HOG(img2)
        distances = distance_HOG(HOG_img1,HOG_img2)
        hog_index.append(i)
        hog_img.append(distances)
        hog_class.append(img2_class)
        hog_new_class.append(new_class)
        
    #print(cld_img)
    test_dict = {'index':hog_index,'distance':hog_img,'original class':hog_class,'new class':hog_new_class} #把除了0.jpg之外的图片的编号、距离、类别（每一百个为一个类，
    #如0-99为一类），是否为同一类（是为1，否为0）
    df = pd.DataFrame(test_dict)
    # print(df)
    df.sort_values(by="distance",axis=0,ascending=True,inplace=True) #按照distance进行升序排列,排序后distance小的就是系统以为是和样本同类的
    print(df)
    #print(df["index"])
    
    flag = 0 #定义在new class中遇到的1的个数
    j = 0 #循环到第几个
    precision = []
    recall = []
    
    for c in df['new class']:
        if j== 99:
            break
        if c == 1:
            flag+=1 #当new class为1时flag+1；统计循环j次时的flag
            #print("flag",flag)
        j+=1
        pre = flag/j #precision = 真的预测为真/（真的预测为真+假的预测为真）
        rec = flag/99 #recall = 真的预测为真/（真的预测为真+真的预测为假）
        precision.append(pre) #把获得的pre加入到数组precision中
        recall.append(rec)
        
    # print("precision",precision)
    # print("recall",recall)
    drawline(recall,precision)

