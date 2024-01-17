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
from pypylon import pylon
from imageio import get_writer
import pandas as pd

class ObjectTracker:
    def __init__(self):
        self.initialize_kalman_filter()
        self.initialize_camera()

        self.valxp = [0 for _ in range(55)]
        self.valyp = [0 for _ in range(55)]
        self.prev_idx = None

        self.rospy_init()

    def initialize_kalman_filter(self):
        self.kf = cv2.KalmanFilter(4, 2)
        # ... (Initialize Kalman filter parameters)

    def initialize_camera(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # ... (Configure camera settings)

    def rospy_init(self):
        rospy.init_node('talker', anonymous=True)
        self.rate = rospy.Rate(20)
        self.imgtopic = '/img'
        self.pub = rospy.Publisher(self.imgtopic, Float64MultiArray, queue_size=10)

    def find_nearest_with_cost(self, c_time, spirx, spiry, pixelx, pixely):
        # ... (Your find_nearest_with_cost logic)

    def process_frame(self, img):
        # ... (Your frame processing logic)

    def track_object(self, img):
        # ... (Your object tracking logic)

    def publish_data(self, value):
        layout = MultiArrayLayout([MultiArrayDimension("", 3, 0)], 0)
        datab = Float64MultiArray(layout, value)
        self.pub.publish(datab)

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
            img = self.process_frame(frame)
            self.track_object(img)
            self.publish_data(self.get_data())
            self.rate.sleep()

            if cv2.waitKey(1) == 27:
                break

            grab_result.Release()

        self.camera.StopGrabbing()
        cv2.destroyAllWindows()

    def convert_image(self, grab_result):
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        image = converter.Convert(grab_result)
        return image.GetArray()

    def get_data(self):
        # ... (Your data extraction logic)


if __name__ == "__main__":
    tracker = ObjectTracker()
    tracker.run()














