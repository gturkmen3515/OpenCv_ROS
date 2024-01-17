# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: gat06
"""
#!/usr/bin/env python

import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
import rospy
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from time import time
from pypylon import pylon
from imageio import get_writer
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

class ObjectTracker:
    def __init__(self):
        self.initialize_kalman_filter()
        self.initialize_camera()
        self.initialize_ros()

    def initialize_kalman_filter(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 0.1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 0.1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 0.1
        self.kf.statePost = 1 * np.random.randn(4, 2)

    def initialize_camera(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def initialize_ros(self):
        rospy.init_node('talker', anonymous=True)
        self.rate = rospy.Rate(20)
        self.img_topic = '/img'
        self.pub = rospy.Publisher(self.img_topic, Float64MultiArray, queue_size=10)

    def find_nearest_with_cost(self, c_time, spir_x, spir_y, pixel_x, pixel_y, prev_idx=None):
        dist2 = (spir_x - pixel_x)**2 + (spir_y - pixel_y)**2
        if prev_idx is None:
            idx = np.argsort(dist2.flatten())[0]
        else:
            past_dist2 = (spir_x - int(spir_x[prev_idx]))**2 + (spir_y - int(spir_y[prev_idx]))**2
            weighted_dist2 = dist2 + 10 * past_dist2
            closest3 = np.argsort(weighted_dist2.flatten())[:3]
            costs = abs(closest3 - prev_idx)
            idx_temp = costs.argmin()
            idx = closest3[idx_temp]
        return idx

    def run(self):
        previous = time()
        writer = get_writer(
            'output.avi',
            codec='libx264',
            quality=None,
            ffmpeg_params=['-preset', 'fast', '-crf', '24']
        )

        while not rospy.is_shutdown():
            current = time() - previous
            grab_result = self.camera.RetrieveResult(10, pylon.TimeoutHandling_ThrowException)
            frame = self.convert_image(grab_result)
            img = self.process_frame(frame, current)
            self.publish_data(img, current)
            self.rate.sleep()

            if cv2.waitKey(1) == 27:
                break

            grab_result.Release()

        self.camera.StopGrabbing()
        cv2.destroyAllWindows()

    def convert_image(self, grab_result):
        image = self.converter.Convert(grab_result)
        return image.GetArray()

    def process_frame(self, img, current_time):
        # Your frame processing logic goes here
        # ...

        return img  # You may need to modify this

    def publish_data(self, img, current_time):
        # Your data extraction and publishing logic goes here
        # ...

    def get_data(self):
        # Your data extraction logic goes here
        # ...

if __name__ == "__main__":
    tracker = ObjectTracker()
    tracker.run()














