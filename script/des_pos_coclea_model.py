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
import coclea_model_class5

if __name__=='__main__':
    try:  
        rospy.init_node('cochlea_robot_des')


        r=rospy.Rate(20)
        topic='/cochlea_robot_des'
        pub=rospy.Publisher(topic,Float64MultiArray,queue_size=10)
        i = 0

        while not rospy.is_shutdown():

            data=coclea_model_class5.model(i)
            val=np.array(data.T)
            print(val,i)
            layout = MultiArrayLayout([MultiArrayDimension("", 7, 0)], 0) 
            msg=Float64MultiArray(layout, val)
            pub.publish(msg)
            r.sleep()
            i=i+1
            if i >= 5180:
                rospy.loginfo("Done.")
                break
            else:
                r.sleep()
    except rospy.ROSInterruptException:
        #dst_thread.join()
        rospy.loginfo("node terminated.")
        pass


    
