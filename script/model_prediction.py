# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 01:31:11 2022

@author: atakan
"""
from os.path import join, expanduser
from math import sqrt

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import joblib

veriler = pd.read_excel('ml/data_function.xlsx')
data_raw=np.float64(veriler)

#mr_pos=data[:,1:3]

d_mr_pos = np.gradient(data_raw[:,1:3], axis=0) * 20
# get idx where velocity is higher than tol
tol = 0.001
d_mr_pos_norm = np.linalg.norm(d_mr_pos, axis=1)
valid_idx = np.where(d_mr_pos_norm >= tol)[0]

# only consider valid indexes
data = data_raw[valid_idx,:]
time = data[:,0]- data[0,0]
mr_pos = data[:,1:3]
sum_dis=[]
for i in range(len(mr_pos)):
    val=np.sum(mr_pos[0:i,:],axis=0)
    sum_dis.append(val)
sum_dis=np.float64(sum_dis)

X2 = np.hstack((mr_pos,
               np.linalg.norm(mr_pos, axis=1)[np.newaxis].T,
               sum_dis,
               np.arctan2(mr_pos[:,1], mr_pos[:,0])[np.newaxis].T))


filename3='ml/teta.sav'
model_direct=joblib.load(open(filename3,'rb'))
teta=model_direct.predict(X2)  

#%%
X = np.hstack((mr_pos,
               np.linalg.norm(mr_pos, axis=1)[np.newaxis].T,
               sum_dis,
               np.arctan2(mr_pos[:,1], mr_pos[:,0])[np.newaxis].T,
               teta[np.newaxis].T))
#%%


filename1='ml/man1.sav'
model_direct=joblib.load(open(filename1,'rb'))
man1=model_direct.predict(X)

filename2='ml/man2.sav'
model_direct=joblib.load(open(filename2,'rb'))
man2=model_direct.predict(X)
#%%
# Y1 = np.hstack((np.linalg.norm(m1_pos, axis=1)[np.newaxis].T,
#                np.arctan2(m1_pos[:,1], m1_pos[:,0])[np.newaxis].T%2*np.pi,
#                m1_ang[np.newaxis].T,
#                m1_pos))

# Y2 = np.hstack((np.linalg.norm(m2_pos, axis=1)[np.newaxis].T,
#                 np.arctan2(m2_pos[:,1], m2_pos[:,0])[np.newaxis].T%2*np.pi,
#                 m2_ang[np.newaxis].T,
#                 m2_pos))

plt.subplot(131)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 4)
plt.title("Microrobot Movement")
plt.scatter(mr_pos[:,0], mr_pos[:,1], c="r", s=0.1)
plt.xlabel("X"), plt.ylabel("Y")
plt.gca().set_aspect(1)
# ###
plt.subplot(132)
plt.title("Magnet Movements")
plt.plot(man1[:,3], man1[:,4], c="b", linewidth=0.5)
plt.plot(man2[:,3], man2[:,4], c="g", linewidth=0.5)
plt.xlabel("X"), plt.ylabel("Y")
plt.gca().set_aspect(1)
# ###
plt.subplot(133)
plt.title("Magnet Angles")
plt.plot( time,man1[:,2])
plt.plot( time,man2[:,2])
plt.xlabel("time"), plt.ylabel("radians")
plt.show()
