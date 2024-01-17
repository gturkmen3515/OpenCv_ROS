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
        self.X_ma_var, self.Y_ma_var = self.load_data()
        self.kf = self.setup_kalman_filter()
        self.cap = cv2.VideoCapture(2)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=20, detectShadows=True)
        self.x1_o, self.y1_o, self.x1_p, self.y1_p = 0, 0, 0, 0
        self.valxp, self.valyp = [0 for _ in range(55)], [0 for _ in range(55)]

        rospy.init_node('talker', anonymous=True)
        self.r = rospy.Rate(20)
        imgtopic = '/img'
        self.pub = rospy.Publisher(imgtopic, Float64MultiArray, queue_size=10)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output_coclea.avi', fourcc, 20.0, (640, 480))

    def load_data(self):
        veriler = pd.read_csv('coclea_data2.csv')
        data_ma = np.float64(veriler)
        X_ma = data_ma[:, 7:8] + 10
        Y_ma = data_ma[:, 8:9] + 50
        X_ma_var, Y_ma_var = [], []

        for i in range(1, len(X_ma)):
            numx = np.linspace(int(X_ma[i - 1]), int(X_ma[i]), 100)
            numy = np.linspace(int(Y_ma[i - 1]), int(Y_ma[i]), 100)
            X_ma_var.append(numx)
            Y_ma_var.append(numy)

        return np.reshape(np.float64(X_ma_var), (np.shape(X_ma_var)[0] * np.shape(X_ma_var)[1], 1)), \
               np.reshape(np.float64(Y_ma_var), (np.shape(Y_ma_var)[0] * np.shape(Y_ma_var)[1], 1))

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

    def find_nearest(self, array1, array2, value1, value2):
        array1 = np.asarray(array1)
        idx1 = (np.abs(array1 - value1)).argmin()
        idx2 = (np.abs(array2 - value2)).argmin()
        return idx1, idx2

    def run(self):
        previous = time()
        while not rospy.is_shutdown():
            current = time() - previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)
            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if current <= 2:
                mask = self.process_initial_frames(img)
            else:
                mask = self.process_subsequent_frames(img)

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
                predicted = self.kf.predict()
                x1, x2 = int(predicted[0]), int(predicted[1])
                self.x1_p = (x1) + (x1 - x_m) * 0.0001 + (self.x1_p - self.x1_o) * 1000
                self.y1_p = (x2) + (x2 - y_m) * 0.0001 + (self.y1_p - self.y1_o) * 1000
                self.valxp.append(self.x1_p)
                self.valyp.append(self.y1_p)

            self.process_coordinates()

    def process_initial_frames(self, img):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([60, 0, 0])
        u_b = np.array([255, 255, 60])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=11)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return mask

    def process_subsequent_frames(self, img):
        for k in range(6):
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            l_b = np.array([60, 0, 0])
            u_b = np.array([255, 255, 60])
            mask = cv2.inRange(mask, l_b, u_b)
            mask = cv2.equalizeHist(mask)
            mask = cv2.equalizeHist(mask)

            mask = cv2.medianBlur(mask, 13 + 2 * k)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = self.fgbg.apply(mask)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3 + k, 3 + k)), iterations=7)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) < 2:
                break

        return mask

    def process_coordinates(self):
        x1_pa = np.mean(self.valxp[-30:])
        y1_pa = np.mean(self.valyp[-30:])

        xt = round((0.03849 * float(x1_pa) - 13.779), 1)
        yt = round((0.0388 * float(y1_pa) - 7.255), 1)

        value = (xt, yt)
        layout = MultiArrayLayout([MultiArrayDimension("", 2, 0)], 0)
        datab = Float64MultiArray(layout, value)
        self.pub.publish(datab)
        self.r.sleep()

        cv2.circle(img, (int(358), int(187)), 1, (255, 0, 0), 2)  # origin
        for i in range(0, len(self.X_ma_var), 1):
            cv2.circle(img, (int(self.X_ma_var[i]), int(self.Y_ma_var[i])), 1, (0, 0, 255), 2)

        i1, i2 = self.find_nearest(self.X_ma_var, self.Y_ma_var, self.x1_pa, self.y1_pa)
        cv2.circle(img, (int(self.x1_pa), int(self.y1_pa)), 20, (0, 255, 0), 2)
        cv2.circle(img, (int(self.x1_pa), int(self.y1_pa)), 1, (0, 0, 255), 2)
        cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame', img)
        self.out.write(img)
        self.x1_o = self.x1_p
        self.y1_o = self.y1_p

        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown("Exiting...")
            cv2.destroyAllWindows()
            self.cap.release()
            exit()
        if ret is False:
            print("Can't receive frame (stream end?). Exiting ...")
            rospy.signal_shutdown("Exiting...")
            cv2.destroyAllWindows()
            self.cap.release()
            exit()

if __name__ == "__main__":
    tracker = CocleaTracker()
    tracker.run()
