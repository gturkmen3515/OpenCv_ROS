# -*- coding: utf-8 -*-
"""
Created on Sat Apr 3 14:46:30 2021
@author: gat06
"""

import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
import rospy
from std_srvs.srv import Empty
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from time import time
import pandas as pd

class CocleaVideoTracker:
    def __init__(self):
        # ROS initialization
        rospy.init_node('talker', anonymous=True)
        self.r = rospy.Rate(20)
        self.img_topic = '/img'
        self.pub = rospy.Publisher(self.img_topic, Float64MultiArray, queue_size=10)

        # Video capture
        self.cap = cv2.VideoCapture(2)
        # self.cap = cv2.VideoCapture('coclea_close_10.mp4')
        self.previous = time()

        # Kalman Filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 1 * np.random.randn(4, 2)

    def process_frame(self, img):
        # Placeholder for specific frame processing logic
        # You need to implement the logic based on your requirements
        pass

    def track_object(self):
        while not rospy.is_shutdown():
            current = time() - self.previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Process the frame
            self.process_frame(img)

            if cv2.waitKey(1) == 27:
                break

            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    tracker = CocleaVideoTracker()
    tracker.track_object()
