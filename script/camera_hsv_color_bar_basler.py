# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:11:31 2021

@author: atakan
"""

import cv2
import numpy as np
from pypylon import pylon

class ObjectTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture('coclea_close8.mp4')
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.camera.AcquisitionFrameRate.SetValue(100)
        self.camera.ExposureTime.SetValue(3020)
        
        cv2.namedWindow("Tracking")
        cv2.createTrackbar("LH", "Tracking", 0, 255, self.nothing)
        cv2.createTrackbar("LS", "Tracking", 0, 255, self.nothing)
        cv2.createTrackbar("LV", "Tracking", 0, 255, self.nothing)
        cv2.createTrackbar("UH", "Tracking", 255, 255, self.nothing)
        cv2.createTrackbar("US", "Tracking", 255, 255, self.nothing)
        cv2.createTrackbar("UV", "Tracking", 100, 255, self.nothing)
    
    def nothing(self, x):
        pass

    def start_tracking(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                img = image.GetArray()
                img = cv2.flip(img, 0)
                img = cv2.flip(img, 1)
                scale_percent = 50

                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)

                dsize = (width, height)
                img = cv2.resize(img, dsize)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                l_h = cv2.getTrackbarPos("LH", "Tracking")
                l_s = cv2.getTrackbarPos("LS", "Tracking")
                l_v = cv2.getTrackbarPos("LV", "Tracking")
    
                u_h = cv2.getTrackbarPos("UH", "Tracking")
                u_s = cv2.getTrackbarPos("US", "Tracking")
                u_v = cv2.getTrackbarPos("UV", "Tracking")
    
                l_b = np.array([l_h, l_s, l_v])
                u_b = np.array([u_h, u_s, u_v])
    
                mask = cv2.inRange(hsv, l_b, u_b)
                mask = cv2.equalizeHist(mask)
    
                mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=6)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=9) 

                _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    cv2.drawContours(img, contours, i, (0, 255, 0), 1)
                
                res = cv2.bitwise_and(img, img, mask=mask)
                cv2.imshow("frame", img)
                cv2.imshow("mask", mask)
                cv2.imshow("res", res)
    
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                grab_result.Release()

        self.camera.StopGrabbing()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ObjectTracker()
    tracker.start_tracking()
