#!/usr/bin/env python
from os.path import join, expanduser
import threading

import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import numpy as np
#from time import time


#previous = time()

#data = np.loadtxt(join(expanduser("~"),"Database",'cochlea_des_angles.csv'), delimiter=",", dtype=np.float64)#, skiprows=1)
data = np.loadtxt(join(expanduser("~"),"Database",'desired_0806_1.csv'), delimiter=",", dtype=np.float64)#, skiprows=1)



# GLOBALS
STREAM_DATA = True


def handle_data_stream(req):
    if req.data:
        STREAM_DATA = True
        rospy.loginfo("Started stream.")
    else:
        STREAM_DATA = False
        rospy.loginfo("Paused stream.")


def data_stream_server():
    s = rospy.Service('data_stream_server', Empty, handle_data_stream)
    print("Ready to stream data.")
    rospy.spin()


if __name__=='__main__':
    try:  
        rospy.init_node('cochlea_robot_des')

        #dst_thread = threading.Thread(data_stream_server())
        #dst_thread.start()

        r=rospy.Rate(20)
        topic='/cochlea_robot_des'
        pub=rospy.Publisher(topic,Float64MultiArray,queue_size=10)
        i = 0

        while not rospy.is_shutdown():
            if STREAM_DATA:
    		#current = time()- previous
                val=data[i,:]
                layout = MultiArrayLayout([MultiArrayDimension("", 6, 0)], 0) 
                msg=Float64MultiArray(layout, val)
                pub.publish(msg)
                r.sleep()
                i=i+2
                if i >= data.shape[0]:
                    rospy.loginfo("Done.")
		    msg=Float64MultiArray(layout, data[0,:])
		    pub.publish(msg)
                    break
            else:
                r.sleep()
    except rospy.ROSInterruptException:
        #dst_thread.join()
        rospy.loginfo("node terminated.")
        pass


    
