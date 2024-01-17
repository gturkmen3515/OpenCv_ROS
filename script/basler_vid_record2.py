# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: atakan
"""#!/usr/bin/env python

import cv2
from time import time
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from pypylon import pylon
from imageio import get_writer
import rospy

class CameraImagePublisher:
    def __init__(self, output_filename='output.avi'):
        rospy.init_node('talker', anonymous=True)
        self.img_topic = '/img'
        self.pub = rospy.Publisher(self.img_topic, Float64MultiArray, queue_size=10)

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.camera.ExposureTime.SetValue(10000)
        
        self.writer = get_writer(
            output_filename,
            codec='libx264',
            quality=None,
            ffmpeg_params=[
                '-preset', 'fast',
                '-crf', '24'
            ]
        )

    def publish_image(self, img):
        layout = MultiArrayLayout([MultiArrayDimension("", 3, 0)], 0) 
        value = (round(posx, 3), round(posy, 3), round(float(phi_var[idx]), 3))
        datab = Float64MultiArray(layout, value)
        self.pub.publish(datab)

    def start_publishing(self):
        previous = time()
        while True:
            current = time() - previous
            grab_result = self.camera.RetrieveResult(10, pylon.TimeoutHandling_ThrowException)
            image = self.converter.Convert(grab_result)
            img = image.GetArray()
            
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.putText(img, str(current), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Frame', img)
            self.writer.append_data(img)
            if cv2.waitKey(1) == 27:
                break
            grab_result.Release()
        
        self.camera.StopGrabbing()
        cv2.destroyAllWindows()
        self.writer.close()

if __name__ == "__main__":
    camera_publisher = CameraImagePublisher()
    camera_publisher.start_publishing()
