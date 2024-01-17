#!/usr/bin/env python
from os.path import join, expanduser
import threading

import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import numpy as np
import joblib
#from time import time
import pandas as pd
import coclea_model_class6
#current_phi_val=0
phi_val=[]
rospy.init_node('cochlea_robot_des')
size=0
i=0

def callback(pos_message):
	px=pos_message.data[0]
	py=pos_message.data[1]
	phi=pos_message.data[2]
	phi_val.append(phi)
	#current_phi_val=int(phi_val[0])
	drive(int(phi_val[0]))
def drive(current_phi):
	global i
        r=rospy.Rate(1)
        topic='/cochlea_robot_des'
        pub=rospy.Publisher(topic,Float64MultiArray,queue_size=10)
        des_phi = -5
    	#current_phi=-2

        while not rospy.is_shutdown():

            data_d=coclea_model_class6.model(i,current_phi,des_phi)
            val=np.array(data_d.T)
	    size=np.shape(val)[1]
            if size == 0 or current_phi==des_phi:
                rospy.loginfo("Done.")
		rospy.signal_shutdown("kapa")
            else:
                r.sleep()
	        print(current_phi,val,i)
	        layout = MultiArrayLayout([MultiArrayDimension("", 5, 0)], 0) 
	        msg=Float64MultiArray(layout, val)
	        pub.publish(msg)
	    	i=i+1


if __name__=='__main__':
		rospy.Subscriber('/img',Float64MultiArray,callback)
		rospy.spin()



    
