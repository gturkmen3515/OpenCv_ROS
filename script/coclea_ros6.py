# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: gat06
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: gat06
"""
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
import rospy
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from time import time
import pandas as pd

class CocleaTracker:
    def __init__(self):
        self.X_ma_var, self.Y_ma_var, self.phi_var = self.load_data()
        self.kf = self.setup_kalman_filter()
        self.cap = cv2.VideoCapture(3)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=300, detectShadows=True)
        self.x1_o, self.y1_o, self.x1_p, self.y1_p = 0, 0, 0, 0
        self.prev = None

        rospy.init_node('talker', anonymous=True)
        self.r = rospy.Rate(20)
        imgtopic = '/img'
        self.pub = rospy.Publisher(imgtopic, Float64MultiArray, queue_size=10)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output_coclea.avi', fourcc, 20.0, (640, 480))

    def load_data(self):
        veriler = pd.read_csv('data/coclea_data2.csv')
        data_ma = np.float64(veriler)
        phi = data_ma[:, 6:7]
        X_ma = data_ma[:, 7:8] + 10
        Y_ma = data_ma[:, 8:9] + 20

        X_ma_var, Y_ma_var, phi_var = [], [], []
        for i in range(1, len(X_ma)):
            num_x = np.linspace(int(X_ma[i - 1]), int(X_ma[i]), 100)
            num_y = np.linspace(int(Y_ma[i - 1]), int(Y_ma[i]), 100)
            num_phi = np.linspace(float(phi[i - 1]), float(phi[i]), 100)
            X_ma_var.append(num_x)
            Y_ma_var.append(num_y)
            phi_var.append(num_phi)

        return (
            np.reshape(np.float64(X_ma_var), (np.shape(X_ma_var)[0] * np.shape(X_ma_var)[1], 1)),
            np.reshape(np.float64(Y_ma_var), (np.shape(Y_ma_var)[0] * np.shape(Y_ma_var)[1], 1)),
            np.reshape(np.float64(phi_var), (np.shape(phi_var)[0] * np.shape(phi_var)[1], 1))
        )

    def setup_kalman_filter(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1
        kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        kf.statePost = 1 * np.random.randn(4, 2)
        return kf

    def find_nearest_with_cost(self, c_time, spirx, spiry, pixelx, pixely, prev_idx=None):
        dist2 = (spirx - pixelx) ** 2 + (spiry - pixely) ** 2
        if prev_idx is None or c_time < 5:
            idx = np.argsort(dist2.flatten())[0]
        else:
            past_dist2 = (spirx - int(spirx[prev_idx])) ** 2 + (spiry - int(spiry[prev_idx])) ** 2
            weighted_dist2 = dist2 + 5 * past_dist2
            closest3 = np.argsort(weighted_dist2.flatten())[:3]
            costs = abs(closest3 - prev_idx)
            idx_temp = costs.argmin()
            idx = closest3[idx_temp]
        return idx

    def process_frames(self, img, current):
        if current <= 1:
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            l_b = np.array([60, 0, 0])
            u_b = np.array([255, 255, 60])
            mask = cv2.inRange(mask, l_b, u_b)
            mask = cv2.equalizeHist(mask)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=11)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        else:
            for k in range(6):
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                l_b = np.array([60, 0, 0])
                u_b = np.array([255, 255, 60])
                mask = cv2.inRange(mask, l_b, u_b)
                mask = cv2.equalizeHist(mask)
                mask = cv2.equalizeHist(mask)

                mask = cv2.medianBlur(mask, 13 + 2 * k)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                mask = self.fgbg.apply(mask)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5 + k, 5 + k)), iterations=7)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) < 2:
                    break

        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) > 0:
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            area = cv2.contourArea(cnt)
            (x_m, y_m), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x_m), int(y_m))
            radius = int(radius)

            measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
            self.kf.correct(measured)
            (predicted,) = self.kf.predict()
            x1 = int(predicted[0])
            x2 = int(predicted[1])
            self.x1_p = (x1) + (x1 - x_m) * 0.0001 + (self.x1_p - self.x1_o) * 1000
            self.y1_p = (x2) + (x2 - y_m) * 0.0001 + (self.y1_p - self.y1_o) * 1000
            return self.x1_p, self.y1_p

    def run(self):
        while not rospy.is_shutdown():
            current = time() - previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            x1_pa, y1_pa = self.process_frames(img, current)
            cv2.circle(img, (346, 168), 1, (255, 0, 0), 2)  # origin
            idx = self.find_nearest_with_cost(current, self.X_ma_var, self.Y_ma_var, x1_pa, y1_pa, prev_idx=self.prev)
            self.prev = idx

            for j in range(8):
                cv2.line(img, (346, 168), (int(346 - 350 * np.cos(j * np.pi / 4)), int(158 + 350 * np.sin(j * np.pi / 4))),
                         (255, 255, 0), 1)

            for i in range(0, len(self.X_ma_var), 1):
                cv2.circle(img, (int(self.X_ma_var[i]), int(self.Y_ma_var[i])), 1, (0, 0, 255), 2)

            cv2.circle(img, (int(self.X_ma_var[idx]), int(self.Y_ma_var[idx])), 20, (0, 255, 0), 2)
            cv2.circle(img, (int(x1_pa), int(y1_pa)), 1, (0, 0, 255), 2)
            x1_o, y1_o = self.x1_p, self.y1_p

            xt = round((0.0385 * float((self.X_ma_var[idx])) - 13.317), 1)
            yt = round((0.0388 * float((self.Y_ma_var[idx])) - 6.518), 1)

            value = (xt, yt, round(float(self.phi_var[idx]), 3))
            layout = MultiArrayLayout([MultiArrayDimension("", 3, 0)], 0)
            datab = Float64MultiArray(layout, value)
            self.pub.publish(datab)
            self.r.sleep()

            cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Frame', img)
            cv2.imshow('mask', mask)
            self.out.write(img)

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
	cap.release()
	out.release()













