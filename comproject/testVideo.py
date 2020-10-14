# -*- coding: utf-8 -*-
# @TIME     : 2020/9/26 19:13
# @Author   : Chen Shan
# @Email    : jacobon@foxmail.com
# @File     : testVideo.py
# @Software : PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取视频
cap = cv2.VideoCapture(r"Siri.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("can not read this video, Exit...")
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', img_gray)

    # if cv2.waitKey(100)& 0xFF ==27:
        # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()


