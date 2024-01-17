#!/usr/bin/env python
from os.path import join, expanduser
import threading
import numpy as np
import joblib
#from time import time
import pandas as pd
def model (i):
	veriler = pd.read_csv('coclea_data2.csv')
	data_raw=np.float64(veriler)



	data_ma=data_raw[:,9:11]
	xt = data_ma[:,0:1]
	yt = data_ma[:,1:2]
	mr_pos = np.hstack((xt,
		            yt))

	sum_dis=[]
	for j in range(len(mr_pos)):
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

	X = np.hstack((mr_pos,
		       np.linalg.norm(mr_pos, axis=1)[np.newaxis].T,
		       sum_dis,
		       np.arctan2(mr_pos[:,1], mr_pos[:,0])[np.newaxis].T,
		       teta[np.newaxis].T))

	filename1='ml/man1.sav'
	model_direct=joblib.load(open(filename1,'rb'))
	man1=model_direct.predict(X)

	filename2='ml/man2.sav'
	model_direct=joblib.load(open(filename2,'rb'))
	man2=model_direct.predict(X)
	data_val = np.hstack((man2[:,3:5],man2[:,2:3],man1[:,3:5],man1[:,2:3],teta[np.newaxis].T))
	print(np.shape(data_val[i:i+1,:]))
	return data_val[i:i+1,:]    
