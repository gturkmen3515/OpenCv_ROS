# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: gat06
"""
#!/usr/bin/env python

import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
from std_msgs.msg import Float64MultiArray
import rospy
from std_srvs.srv import Empty
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from math import atan2, cos, sin, sqrt, pi
from time import time
import pandas as pd
from pypylon import pylon
from imageio import get_writer
def find_nearest_with_cost(c_time,spirx, spiry, pixelx, pixely, prev_idx=None):
    """
    Parameters
    ----------
    spirx : np.ndarray
        DESCRIPTION.
    spiry : np.ndarray
        DESCRIPTION.
    pixelx : int
        DESCRIPTION.
    pixely : int
        DESCRIPTION.
    prev_idx : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    int
        DESCRIPTION.
    """
    dist2 = (spirx - pixelx)**2 + (spiry - pixely)**2
    if prev_idx==None:
        idx = np.argsort(dist2.flatten())[0]
    else:
        # Calculate cost
        past_dist2 = (spirx - int(spirx[prev_idx]))**2 + (spiry - int(spiry[prev_idx]))**2
        weighted_dist2 = dist2 + 5*past_dist2
        closest3 = np.argsort(weighted_dist2.flatten())[:3]
        costs = abs(closest3 - prev_idx)
        # Select idx from cost
        idx_temp = costs.argmin()
        idx = closest3[idx_temp]
    return idx

points=[]


#veriler = pd.read_csv('data/coclea_data2.csv')
veriler = pd.read_csv('data/coclea_data_road_final3.csv')

data_ma=np.float64(veriler)
phi = data_ma[:,2:3]
X_ma = data_ma[:,0:1]
Y_ma = data_ma[:,1:2]
#Z_ma = data_ma[:,5:6]
#teta=data[:,6:7]
previous = time()
X_ma_var=[]
Y_ma_var=[]
phi_var=[]
for i in range(1,len(X_ma)):
    num_x=np.linspace(int(X_ma[i-1]),int(X_ma[i]),100)
    num_y=np.linspace(int(Y_ma[i-1]),int(Y_ma[i]),100)
    num_phi=np.linspace(float(phi[i-1]),float(phi[i]),100)
    X_ma_var.append(num_x)
    Y_ma_var.append(num_y)
    phi_var.append(num_phi)

X_ma_var=np.reshape(np.float64(X_ma_var),(np.shape(X_ma_var)[0]*np.shape(X_ma_var)[1],1))
Y_ma_var=np.reshape(np.float64(Y_ma_var),(np.shape(Y_ma_var)[0]*np.shape(Y_ma_var)[1],1))
phi_var=np.reshape(np.float64(phi_var),(np.shape(phi_var)[0]*np.shape(phi_var)[1],1))

kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.measurementNoiseCov=np.array([[1, 1], [1,1]], np.float32) * 1
kf.errorCovPre= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.errorCovPost= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) *1
kf.statePost = 1 * np.random.randn(4, 2)


#fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.bgsegm.BackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2(history = 5000,varThreshold = 1,detectShadows=True)
#fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)



prev = None

data_val=[]
center = (int(X_ma[0:1]),int(Y_ma[0:1]))
cur_old=0
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

#out = cv2.VideoWriter('output_coclea.avi', fourcc, 20.0, (1280,  720))
#(1280,720)
valxp=[0 for i in range(55)]
valyp=[0 for i in range(55)]

rospy.init_node('talker', anonymous=True)
r=rospy.Rate(20)
imgtopic='/img'
pub=rospy.Publisher(imgtopic,Float64MultiArray,queue_size=10)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(100)
camera.ExposureTime.SetValue(3010)
writer = get_writer(
       'output-filename.mkv',  # mkv players often support H.264
        codec='libx264',  # When used properly, this is basically "PNG for video" (i.e. lossless)
        quality=None,  # disables variable compression
        ffmpeg_params=[  # compatibility with older library versions
            '-preset',   # set to fast, faster, veryfast, superfast, ultrafast
            'fast',      # for higher speed but worse compression
            '-crf',      # quality; set to 0 for lossless, but keep in mind
            '24'         # that the camera probably adds static anyway
        ]
)

x_m=0
y_m=0

x1_o=0
y1_o=0

x1_p=0
y1_p=0
while not rospy.is_shutdown():
    current = time()- previous
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    # Access the image data
    image = converter.Convert(grabResult)
    frame = image.GetArray()
    img=cv2.flip(frame, 0)
    img=cv2.flip(img, 1)
    scale_percent = 50

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)    )
    if current <=5:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 0, 0])
        u_b = np.array([255, 255, 120])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=4)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=11)
        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if current>5:
        #for k in range(6):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 0, 0])
        u_b = np.array([255, 255, 120])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=2)
        #mask = cv2.medianBlur(mask, 3+2*k)

        #mask = cv2.equalizeHist(mask)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=5)

        #mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
        #mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(5+k,5+k)), iterations=2)
        mask = fgbg.apply(mask)

        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #if len(contours) < 2 :
            #break
    
    areas = [cv2.contourArea(c) for c in contours]
    if  len(areas)>0:
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        area = cv2.contourArea(cnt)
        (x_m,y_m),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x_m),int(y_m))
        radius = int(radius)
    
    measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
    kf.correct(measured)
    (predicted) = kf.predict()
    x1=int(predicted[0])
    x2=int(predicted[1])
    x1_p=(x1)+(x1-x_m)*0+(x1_p-x1_o)*0
    y1_p=(x2)+(x2-y_m)*0+(y1_p-y1_o)*0
    valxp.append(x1_p)
    valyp.append(y1_p)
    x1_pa=np.mean(valxp[-30:])
    y1_pa=np.mean(valyp[-30:])
    idx = find_nearest_with_cost(current,X_ma_var,Y_ma_var, x1_pa, y1_pa, prev_idx=prev)
    prev=idx
    orijin=(495,250)
    cv2.circle(img,((orijin[0],orijin[1])),1,(255,0,0),2)#origin
    for j in range(8):
        cv2.line(img, (orijin[0],orijin[1]), (int(orijin[0]-orijin[0]*np.cos(j*np.pi/4)),int(orijin[1]+orijin[0]*np.sin(j*np.pi/4))), (0,255,0),2)
    for i in range(0,len(X_ma_var),1):
        cv2.circle(img,(int(X_ma_var[i]),int(Y_ma_var[i])),1,(0,0,255),2)
        cv2.circle(img,(int(X_ma_var[idx ]),int(Y_ma_var[idx ])),20,(0,255,0),2)
    cv2.circle(img,(int(x1_pa),int(y1_pa)),1,(0,0,255),2)
    x1_o=x1_pa
    y1_o=y1_pa
    xt=round((0.0385*float((X_ma_var[idx]))-18.711),1)
    yt=round((0.0388*float((Y_ma_var[idx]))-10.2044),1)
    
    #xt=round((0.03849*float(x1_pa)-13.47),1)
    #yt=round((0.0388*float(y1_pa)-5.819),1)
    value=(xt,yt,round(float(phi_var[idx]),3))
    layout = MultiArrayLayout([MultiArrayDimension("", 3, 0)], 0) 
    datab=Float64MultiArray(layout, value)
    pub.publish(datab)
    r.sleep()
    cv2.putText(img, str(value), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', img)
    cv2.imshow('mask', mask)
    #vid_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #out.write(img)
    writer.append_data(img)
    #res.Release()
    if  cv2.waitKey(1) == 27:
        break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()
#out.release()













