# -*- coding: utf-8 -*-
"""
Created on Sat Apr 3 14:46:30 2021
@author: atakan
"""

import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
import rospy
from std_srvs.srv import Empty
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from time import time
import pandas as pd

class CocleaTracker:
    def __init__(self):
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 1 * np.random.randn(4, 2)

        # Video capture
        self.cap = cv2.VideoCapture(2)
        #self.cap = cv2.VideoCapture('coclea_close_10.mp4')
        self.previous = time()

        # Background subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=5, detectShadows=True)

        # Previous coordinates
        self.x1_o, self.y1_o, self.x1_p, self.y1_p = 0, 0, 0, 0

        # ROS initialization
        rospy.init_node('talker', anonymous=True)
        self.r = rospy.Rate(20)
        self.img_topic = '/img'
        self.pub = rospy.Publisher(self.img_topic, Float64MultiArray, queue_size=10)

    def process_frame(self, img):
        # Your frame processing logic goes here
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
    tracker = CocleaTracker()
    tracker.track_object()
