# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 01:31:11 2022

@author: atakan
"""
import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils

from math import atan2, cos, sin, sqrt, pi
from time import time
import pandas as pd

def find_nearest_with_cost(spirx, spiry, pixelx, pixely, prev_idx=None):
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


veriler = pd.read_csv('coclea_data2.csv')

data_ma=np.float64(veriler)
X_ma = data_ma[:,7:8]+10
Y_ma = data_ma[:,8:9]+20
#Z_ma = data_ma[:,5:6]
#teta=data[:,6:7]
previous = time()
X_ma_var=[]
Y_ma_var=[]
for i in range(1,len(X_ma)):
    numx=np.linspace(int(X_ma[i-1]),int(X_ma[i]),100)
    numy=np.linspace(int(Y_ma[i-1]),int(Y_ma[i]),100)
    X_ma_var.append(numx)
    Y_ma_var.append(numy)

X_ma_var=np.reshape(np.float64(X_ma_var),(np.shape(X_ma_var)[0]*np.shape(X_ma_var)[1],1))
Y_ma_var=np.reshape(np.float64(Y_ma_var),(np.shape(Y_ma_var)[0]*np.shape(Y_ma_var)[1],1))

data=np.concatenate((X_ma_var,Y_ma_var),axis=1)

np.savetxt("coclea_road_large.csv", 
           data,
           delimiter =", ", 
           fmt ='% s')