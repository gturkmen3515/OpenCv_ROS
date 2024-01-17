# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:40:45 2022

@author: atakan
"""
import cv2
import numpy as np
from time import time
import joblib
import pandas as pd

class CocleaTracker:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 1 * np.random.randn(4, 2)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=5, detectShadows=True)
        self.previous_time = time()

    def process_frame(self, frame):
        current_time = time() - self.previous_time
        img = cv2.flip(frame, 0)
        img = cv2.flip(img, 1)

        if current_time <= 2:
            mask = self._create_mask(img)
            contours = self._find_contours(mask)
        else:
            for k in range(6):
                mask = self._create_mask(img)
                mask = self._apply_morphological_operations(mask, k)
                contours = self._find_contours(mask)
                if len(contours) < 2:
                    break

        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) > 0:
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            measured = np.array([[np.float32(cnt[:, 0].mean())], [np.float32(cnt[:, 1].mean())]])
            self.kf.correct(measured)
            predicted = self.kf.predict()
            x1 = int(predicted[0])
            x2 = int(predicted[1])

            xt, yt = self._calculate_coordinates(x1, x2)
            value = (xt, yt)
            self._draw_circles_and_text(img, cnt, x1, x2, value)

            diffx = (x1 - self.x1_o) / (current_time - self.cur_old)
            diffy = (x2 - self.y1_o) / (current_time - self.cur_old)
            val_full = (xt, yt, current_time, diffx, diffy)
            self.data_val.append(val_full)

            self.x1_o = x1
            self.y1_o = x2
            self.cur_old = current_time

    def _create_mask(self, img):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([60, 0, 0])
        u_b = np.array([255, 255, 60])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=11)
        return mask

    def _apply_morphological_operations(self, mask, k):
        mask = cv2.medianBlur(mask, 15)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = self.fgbg.apply(mask)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3 + k, 3 + k)), iterations=7)
        return mask

    def _find_contours(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    def _calculate_coordinates(self, x1, x2):
        x1_p = (x1) + (x1 - self.x1_o) * 0.0001 + (x1_p - self.x1_o) * 10000
        y1_p = (x2) + (x2 - self.y1_o) * 0.0001 + (y1_p - self.y1_o) * 10000
        valxp.append(x1_p)
        valyp.append(y1_p)

        x1_pa = np.mean(valxp[-20:])
        y1_pa = np.mean(valyp[-20:])
        xt = round((0.03849 * float(x1_pa) - 13.47), 1)
        yt = round((0.0388 * float(y1_pa) - 5.819), 1)
        return xt, yt

    def _draw_circles_and_text(self, img, cnt, x1, x2, value):
        center = (int(x1), int(x2))
        radius = int(cv2.minEnclosingCircle(cnt)[1])
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        cv2.circle(img, center, 1, (255, 0, 0), 2)
        cv2.circle(img, (int(x1_pa), int(y1_pa)), 20, (0, 255, 0), 2)
        cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret == False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.process_frame(frame)

            cv2.imshow('FG MASK Frame', mask)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = CocleaTracker(video_source=0)
    tracker.run()
