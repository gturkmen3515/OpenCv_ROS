# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: atakan
"""
#!/usr/bin/env python
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
import rospy
from time import time

class ObjectTracker:
    def __init__(self):
        self.previous = time()
        self.kf = cv2.KalmanFilter(4, 2)
        self.kalman_filter_setup()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.valxp = [0 for _ in range(5)]
        self.valyp = [0 for _ in range(5)]
        self.center = (0, 0)
        self.x1_o = 0
        self.y1_o = 0
        self.x1_p = 0
        self.y1_p = 0

        rospy.init_node('talker', anonymous=True)
        self.r = rospy.Rate(20)
        self.imgtopic = '/img'
        self.pub = rospy.Publisher(self.imgtopic, Float64MultiArray, queue_size=10)

        self.cap = cv2.VideoCapture(2)
        self.out = cv2.VideoWriter('output_coclea.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    def kalman_filter_setup(self):
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-5
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1e-3
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 0.01 * np.random.randn(4, 2)

    def apply_color_filter(self, img, k):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([60 - k * 6, 0, 0])
        u_b = np.array([255, 255, 60 + k * 6])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.equalizeHist(mask)

        mask = cv2.medianBlur(mask, 21 + k * 12)
        mask = cv2.GaussianBlur(mask, (21 + k * 12, 21 + k * 12), 0)
        mask = self.fgbg.apply(mask)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (4 + k, 4 + k)), iterations=9)
        return mask

    def track_object(self, img, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) > 0:
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            area = cv2.contourArea(cnt)
            (x_m, y_m), radius = cv2.minEnclosingCircle(cnt)
            self.center = (int(x_m), int(y_m))
            radius = int(radius)

            measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
            self.kf.correct(measured)
            predicted = self.kf.predict()
            x1 = int(predicted[0])
            x2 = int(predicted[1])
            self.x1_p = (x1) + (x1 - x_m) * 0.0001 + (self.x1_p - self.x1_o) * 1000
            self.y1_p = (x2) + (x2 - y_m) * 0.0001 + (self.y1_p - self.y1_o) * 1000
            self.valxp.append(self.x1_p)
            self.valyp.append(self.y1_p)

    def publish_data(self):
        x1_pa = np.mean(self.valxp[-3:])
        y1_pa = np.mean(self.valyp[-3:])
        xt = round((0.03422 * float(x1_pa) - 11.73), 1)
        yt = round((0.03279 * float(y1_pa) - 6.46), 1)
        value = (xt, yt)
        layout = MultiArrayLayout([MultiArrayDimension("", 2, 0)], 0)
        datab = Float64MultiArray(layout, value)
        self.pub.publish(datab)
        self.r.sleep()

    def run(self):
        while not rospy.is_shutdown():
            current = time() - self.previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)
            if ret == False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            mask = None
            if current <= 2:
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                l_b = np.array([60, 0, 0])
                u_b = np.array([255, 255, 60])
                mask = cv2.inRange(mask, l_b, u_b)
                mask = cv2.equalizeHist(mask)
                mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=5)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=11)
            elif current > 2:
                for k in range(6):
                    mask = self.apply_color_filter(img, k)
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    if len(contours) < 2:
                        break
            self.track_object(img, mask)
            self.publish_data()

            cv2.circle(img, (int(343), int(197)), 1, (255, 0, 0), 2)  # origin
            cv2.circle(img, (int(self.center[0]), int(self.center[1])), 1, (0, 0, 255), 2)
            cv2.circle(img, (int(self.center[0]), int(self.center[1])), 20, (0, 255, 0), 2)
            cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Frame', img)
            self.out.write(img)

            self.x1_o = self.x1_p
            self.y1_o = self.y1_p

            if cv2.waitKey(1) == 27:
                break
            if ret == False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    tracker = ObjectTracker()
    tracker.run()
