# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:40:45 2022

@author: atakan
"""
import cv2
import numpy as np
from time import time
from pypylon import pylon

class Tracker:
    def __init__(self):
        self.previous = time()
        self.kf = cv2.KalmanFilter(4, 2)
        self.kalman_filter_setup()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.valxp = [0 for _ in range(5)]
        self.valyp = [0 for _ in range(5)]
        self.center = (0, 0)
        self.cap = cv2.VideoCapture(2)
        self.out = cv2.VideoWriter('output_coclea.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    def kalman_filter_setup(self):
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-5
        self.kf.measurementNoiseCov = 1e-1 * np.array([[1, 1], [1, 1]], np.float32)
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 0.1 * np.random.randn(4, 2)

    def apply_color_filter(self, img, k):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([40 + k * 6, 0, 0])
        u_b = np.array([255, 255, 80 - k * 6])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.medianBlur(mask, 21 + k * 10)
        mask = cv2.GaussianBlur(mask, (21 + k * 12, 21 + k * 10), 0)
        mask = self.fgbg.apply(mask)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5 + k, 5 + k)), iterations=8)
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
            measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
            self.kf.correct(measured)
            predicted = self.kf.predict()
            x1 = int(predicted[0])
            x2 = int(predicted[1])
            x1_p = (x1) + (x1 - x_m) * 0.00001 + (x1_p - x1_o) * 1000
            y1_p = (x2) + (x2 - y_m) * 0.00001 + (y1_p - y1_o) * 1000
            self.valxp.append(x1_p)
            self.valyp.append(y1_p)
        return self.center, radius

    def compute_position(self):
        x1_pa = np.mean(self.valxp[-5:])
        y1_pa = np.mean(self.valyp[-5:])
        xt = round((0.03422 * float(x1_pa) - 13.24), 1)
        yt = round((0.03279 * float(y1_pa) - 7.71), 1)
        return xt, yt

    def run(self):
        while True:
            current = time() - self.previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)

            if ret:
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

                center, radius = self.track_object(img, mask)
                xt, yt = self.compute_position()

                cv2.circle(img, center, 1, (0, 0, 255), 2)
                x1_o, y1_o = x1_p, y1_p  # update previous values
                cv2.circle(img, center, 20, (0, 255, 0), 2)
                cv2.putText(img, str((xt, yt)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                cv2.imshow('FG MASK Frame', mask)
                cv2.imshow('Frame', img)
                self.out.write(img)

                if cv2.waitKey(1) == 27:
                    break
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()


