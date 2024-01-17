# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:40:45 2022

@author: atakan
"""


from matplotlib import pyplot as plt
import numpy as np
import cv2
from time import time
import joblib
import pandas as pd


class CocleaTracker:
    def __init__(self, data_file='data/coclea_data_road.xlsx', video_file='coclea_close_10.mp4'):
        self.veriler = pd.read_excel(data_file)
        self.data_ma = np.float64(self.veriler)
        self.X_ma = self.data_ma[:, 7:8]
        self.Y_ma = self.data_ma[:, 8:9]
        self.X_ma_var = []
        self.Y_ma_var = []

        for i in range(1, len(self.X_ma)):
            numx = np.linspace(int(self.X_ma[i - 1]), int(self.X_ma[i]), 100)
            numy = np.linspace(int(self.Y_ma[i - 1]), int(self.Y_ma[i]), 100)
            self.X_ma_var.append(numx)
            self.Y_ma_var.append(numy)

        self.X_ma_var = np.reshape(np.float64(self.X_ma_var), (np.shape(self.X_ma_var)[0] * np.shape(self.X_ma_var)[1], 1))
        self.Y_ma_var = np.reshape(np.float64(self.Y_ma_var), (np.shape(self.Y_ma_var)[0] * np.shape(self.Y_ma_var)[1], 1))

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1, 1]], np.float32) * 1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 1 * np.random.randn(4, 2)

        self.cap = cv2.VideoCapture(video_file)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=2000, detectShadows=True)

        self.x1_o = 0
        self.y1_o = 0
        self.x1_p = 0
        self.y1_p = 0
        self.data_val = []
        self.center = (int(self.X_ma[0:1]), int(self.Y_ma[0:1]))
        self.cur_old = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output_coclea.avi', self.fourcc, 20.0, (640, 480))
        self.valxp = [0 for i in range(55)]
        self.valyp = [0 for i in range(55)]
        self.teta_val = [0,]
        self.error_tra = 100
        self.i = 0
        self.past = 0
        self.errorx_past = 0
        self.errory_past = 0
        self.prev_val = []
        self.prev = None

    def find_nearest_with_cost(self, spirx: np.ndarray, spiry: np.ndarray, pixelx: int, pixely: int,
                               prev_idx: int = None) -> int:
        dist2 = (spirx - pixelx) ** 2 + (spiry - pixely) ** 2
        if prev_idx is None:
            idx = np.argsort(dist2.flatten())[0]
        else:
            past_dist2 = (spirx - int(spirx[prev_idx])) ** 2 + (spiry - int(spiry[prev_idx])) ** 2
            weighted_dist2 = dist2 + 5 * past_dist2
            closest3 = np.argsort(weighted_dist2.flatten())[:3]
            costs = abs(closest3 - prev_idx)
            idx_temp = costs.argmin()
            idx = closest3[idx_temp]
        return idx

    def run_tracking(self):
        while True:
            current = time() - self.previous
            ret, img = self.cap.read()
            img = cv2.flip(img, 0)
            img = cv2.flip(img, 1)
            dt_ch = current - self.past

            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if current <= 2:
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

                    mask = cv2.medianBlur(mask, 15)
                    mask = cv2.GaussianBlur(mask, (7, 7), 0)
                    mask = self.fgbg.apply(mask)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3 + k, 3 + k)), iterations=7)
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
                predicted = self.kf.predict()
                x1 = int(predicted[0])
                x2 = int(predicted[1])
                self.x1_p = (x1) + (x1 - x_m) * 0.0001 + (self.x1_p - self.x1_o) * 10000
                self.y1_p = (x2) + (x2 - y_m) * 0.0001 + (self.y1_p - self.y1_o) * 10000
                self.valxp.append(self.x1_p)
                self.valyp.append(self.y1_p)

                x1_pa = np.mean(self.valxp[-20:])
                y1_pa = np.mean(self.valyp[-20:])

                xt = round((0.03849 * float(x1_pa) - 13.47), 1)
                yt = round((0.0388 * float(y1_pa) - 5.819), 1)

                value = (xt, yt)

                idx = self.find_nearest_with_cost(self.X_ma_var, self.Y_ma_var, x1_pa, y1_pa, prev_idx=self.prev)
                self.prev = idx

                for i in range(0, len(self.X_ma), 1):
                    cv2.circle(img, (int(self.X_ma[i]), int(self.Y_ma[i])), 1, (0, 0, 255), 2)
                cv2.circle(img, (int(self.X_ma_var[idx]), int(self.Y_ma_var[idx])), 20, (0, 255, 0), 2)
                cv2.circle(img, (int(350), int(150)), 1, (255, 0, 0), 2)

                cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('FG MASK Frame', mask)
                cv2.imshow('Frame', img)
                self.out.write(img)
                diffx = (self.x1_p - self.x1_o) / (current - self.cur_old)
                diffy = (self.y1_p - self.y1_o) / (current - self.cur_old)
                val_full = (xt, yt, current, diffx, diffy)
                self.data_val.append(val_full)
                self.x1_o = self.x1_p
                self.y1_o = self.y1_p
                self.cur_old = current

                if cv2.waitKey(1) == 27:
                    break

            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# Instantiate the CocleaTracker class and run the tracking
coclea_tracker = CocleaTracker()
coclea_tracker.run_tracking()
