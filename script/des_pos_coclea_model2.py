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

if __name__=='__main__':
    try:
  
        rospy.init_node('cochlea_robot_des')
        r=rospy.Rate(20)
        topic='/cochlea_robot_des'
        pub=rospy.Publisher(topic,Float64MultiArray,queue_size=10)
        des_phi = -3
    	current_phi=-2
	i=0
        while not rospy.is_shutdown():

            data_d=coclea_model_class6.model(i,current_phi,des_phi)
            val=np.array(data_d.T)
            if np.shape(val)[1] ==0 :
                rospy.loginfo("Done.")
                break
            else:
                r.sleep()
            print(val,i)
            layout = MultiArrayLayout([MultiArrayDimension("", 5, 0)], 0) 
            msg=Float64MultiArray(layout, val)
            pub.publish(msg)
            r.sleep()
	    i=i+1

    except rospy.ROSInterruptException:
        #dst_thread.join()
        rospy.loginfo("node terminated.")
        pass


    
