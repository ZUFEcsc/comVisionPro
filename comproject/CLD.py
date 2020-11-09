# -*- coding: utf-8 -*-
# @TIME     : 2020/10/26 12:03
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : CLD.py
# @Software : PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import dct

def cvshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def CLD(img):
    #图像分割8*8
    n=8;

    (height,width,channel)=img.shape
    #print(height,width,channel) #256,383,3

    block_h = np.fix(height/n); #每块的高度
    block_w=np.fix(width/n); #每块的宽度
    #print(block_h,block_w) #32.0 48.0

    im_n=np.zeros((n,n,channel))

    for i in range(n):
        for j in range(n):
            for k in range(channel):
            #确定块
                a = block_h * i+1;
                b = block_h * (i+1); #height: b-a
                c = block_w * j+1;
                d = block_w * (j+1); #width: d-c
            #循环到右下角的块时
                if i == (n-1):
                    b = height-1;
                if j == (n-1):
                    d = width-1;
            #每块代表色的选择，实现“mpeg-7标准推荐使用区域块的平均像素颜色值作为代表颜色”
                #print(img[int(a)][int(d)][int(k)])

                arr=[img[int(a)][int(c)][k],img[int(a)][int(d)][k],img[int(b)][int(c)][k],img[int(b)][int(d)][k]]
                
                pix = np.mean(np.mean(arr));
                #print(pix)
                im_n[i][j][k]=pix
                #print(im_n)

    # 将rgb转换色彩空间为YCbCr
    mat = np.array(
       [[ 65.481, 128.553, 24.966 ],
        [-37.797, -74.203, 112.0  ],
        [  112.0, -93.786, -18.214]])
    offset = np.array([16, 128, 128])


    im_YCbCr = rgb2ycbcr(mat,offset,im_n)
    
    #DCT变换
    im_DCT = np.zeros((n,n,channel)); 
    #因为dct操作只能对二维矩阵进行操作，所以这里要把r,g,b分别拎出来处理
    im_DCT[:,:,0] = dct(im_YCbCr[:,:,0])
    im_DCT[:,:,1] = dct(im_YCbCr[:,:,1])
    im_DCT[:,:,2] = dct(im_YCbCr[:,:,2])
    #print(im_DCT)

    #按照之字形扫描im_DCT存储到descript中
    zig = [[0   ,  1  ,   5  ,   6  ,  14  ,  15  ,  27  ,  28],
           [2   ,  4  ,   7  ,  13  ,  16 ,   26  ,  29  ,  42],
           [3   ,  8  ,  12 ,   17  ,  25 ,   30  ,  41  ,  43],
           [9   , 11  ,  18  ,  24  ,  31 ,   40  ,  44  ,  53],
           [10   , 19  ,  23 ,   32  ,  39 ,   45  ,  52  ,  54],
           [20   , 22  ,  33  ,  38  ,  46 ,   51  ,  55  ,  60],
           [21   , 34  ,  37  ,  47  ,  50 ,   56  ,  59  ,  61],
           [35   , 36  ,  48  ,  49  ,  57 ,   58  ,  62  ,  63 ]]
    descript = np.zeros((n*n,channel));
    for i in range (n):
        for j in range (n):
            descript[zig[i][j],:] = im_DCT[i,j,:];
            #print(descript);

    result = descript;
    return result;

#颜色空间转换的函数
def rgb2ycbcr(mat,offset,rgb_img):
    n=8
    channel=3
    ycbcr_img = np.zeros((n,n,channel))
    for x in range(n):
        for y in range(n):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def distance_CLD(img1,img2):
    return np.sqrt(np.sum(np.square(img1 - img2))) #欧式距离

def drawline(recall,precision):
    plt.plot(recall,precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title('PR Graph of CLD')
    plt.show()

if __name__ == "__main__":
    #img2 = cv2.imread('./Corel-1K/image.orig/3.jpg')
    img_index = 0; #设置样本img的编号
    img=cv2.imread('./Corel-1K/image.orig/'+str(img_index)+'.jpg') #这里就是读取0.jpg
    img_class = int((img_index)/100)+1 #算出样本img的类别，0-99为类1,100-199为类2,，以此类推
    print("class",img_class)
    
    CLD_img1 = CLD(img) #对img进行CLD处理
    cld_index=[] 
    cld_img=[]
    cld_class = []
    cld_new_class = [] 
    for i in range(0,1000):
        if i == img_index: #如果读到样本img的编号就跳过
            print("skip>>>>>>>"+str(i))
            continue;
        img2 = cv2.imread('..\\Corel-1K\\image.orig\\%d.jpg' %(i))
        img2 = cv2.resize(img2,(256,383))
        #print(img2.shape)
        #print(i)
        img2_class = int((i)/100)+1
        #print(img2_class)
        if img_class == img2_class: #判断循环取出的图片是否和样本为同一个类别
            new_class = 1
        else:
            new_class = 0
        CLD_img2 = CLD(img2);
        distances = distance_CLD(CLD_img1,CLD_img2);
        cld_index.append(i)
        cld_img.append(distances)
        cld_class.append(img2_class)
        cld_new_class.append(new_class)
        
    #print(cld_img)
    test_dict = {'index':cld_index,'distance':cld_img,'original class':cld_class,'new class':cld_new_class} #把除了0.jpg之外的图片的编号、距离、类别（每一百个为一个类，
    #如0-99为一类），是否为同一类（是为1，否为0）
    df = pd.DataFrame(test_dict)
    #display(df)
    df.sort_values(by="distance",axis=0,ascending=True,inplace=True) #按照distance进行升序排列,排序后distance小的就是系统以为是和样本同类的
    #display(df)
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
        
    #print("precision",precision)
    #print("recall",recall)
    drawline(recall,precision)




