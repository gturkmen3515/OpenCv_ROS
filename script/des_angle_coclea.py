#!/usr/bin/env python
from os.path import join, expanduser
import threading

import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import numpy as np
import time
#import pandas as pd
#veriler = np.loadtxt('desired_val.csv', delimiter=",", skiprows=1)
veriler= np.loadtxt(join(expanduser("~"),"Database",'desired_val.csv'), delimiter=",", dtype=np.float64)#, skiprows=1)
#veriler =np.float64(veriler)




if __name__=='__main__':
    try:  
        rospy.init_node('coclea_publisher')
        r=rospy.Rate(20)
        cocleatopic='/coclea'
        pub=rospy.Publisher(cocleatopic,Float64MultiArray,queue_size=10)
        i = 0
        while not rospy.is_shutdown():
            #val=[cu[i],cv[i],cu_p[i],cv_p[i],rot[i]]
	    val=veriler[i,:]
            layout = MultiArrayLayout([MultiArrayDimension("", 6, 0)], 0) 
            datab=Float64MultiArray(layout, val)
            pub.publish(datab)
            r.sleep()
            i=i+1
            if i >= timev.shape[0]:
                rospy.loginfo("Done.")
                break
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")
        pass


    
