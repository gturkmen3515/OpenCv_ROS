# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""

from pypylon import pylon
import cv2
import numpy as np
import pandas as pd

class CocleaTracker:
    def __init__(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.out = cv2.VideoWriter('output_coclea_basler.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
        self.kf.measurementNoiseCov = np.array([[1, 1], [1,1]], np.float32) * 1
        self.kf.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
        self.kf.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
        self.kf.statePost = 1 * np.random.randn(4, 2)
        self.center = (0, 0)
        self.valxp = [0 for _ in range(55)]
        self.valyp = [0 for _ in range(55)]
        self.prev = None

    def find_nearest_with_cost(self, spirx, spiry, pixelx, pixely, current, prev_idx=None):
        dist2 = (spirx - pixelx)**2 + (spiry - pixely)**2
        if prev_idx is None:
            idx = np.argsort(dist2.flatten())[0]
        else:
            past_dist2 = (spirx - int(spirx[prev_idx]))**2 + (spiry - int(spiry[prev_idx]))**2
            weighted_dist2 = dist2 + 5 * past_dist2
            closest3 = np.argsort(weighted_dist2.flatten())[:3]
            costs = abs(closest3 - prev_idx)
            idx_temp = costs.argmin()
            idx = closest3[idx_temp]
        return idx

    def start_tracking(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = self.process_frame(grab_result)
                self.show_frame(frame)
                if cv2.waitKey(1) == 27:
                    break
            grab_result.Release()

    def process_frame(self, grab_result):
        image = self.converter.Convert(grab_result)
        img = image.GetArray()
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        img = cv2.resize(img, dsize)
        current = time() - self.previous
        dt_ch = current - self.past

        if current <= 5:
            mask = self.create_mask(img)
        else:
            mask = self.create_advanced_mask(img)

        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
            x1_p = x1
            y1_p = x2
            self.valxp.append(x1_p)
            self.valyp.append(y1_p)

        x1_pa = np.mean(self.valxp[-20:])
        y1_pa = np.mean(self.valyp[-20:])

        xt = round((0.03849 * float(x1_pa) - 13.47), 1)
        yt = round((0.0388 * float(y1_pa) - 5.819), 1)

        value = (xt, yt)

        idx = self.find_nearest_with_cost(self.X_ma_var, self.Y_ma_var, x1_pa, y1_pa, current, prev_idx=self.prev)
        self.prev = idx
        origin = (486, 263)
        for j in range(8):
            cv2.line(img, (origin[0], origin[1]), (int(origin[0] - origin[0] * np.cos(j * np.pi / 4)),
                                                   int(origin[1] + origin[0] * np.sin(j * np.pi / 4))), (0, 255, 0), 2)

        for i in range(0, len(self.X_ma), 1):
            cv2.circle(img, (int(self.X_ma[i]), int(self.Y_ma[i])), 1, (0, 0, 255), 2)

        cv2.circle(img, (int(self.X_ma_var[idx]), int(self.Y_ma_var[idx])), 20, (0, 255, 0), 2)
        cv2.circle(img, (origin), 1, (255, 0, 0), 2)

        value = (xt, yt, float(self.phi_ma_var[idx]))
        cv2.putText(img, str(value), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('FG MASK Frame', mask)
        cv2.imshow('Frame', img)
        self.out.write(img)
        self.cur_old = current
        grab_result.Release()
        return img

    def create_mask(self, img):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 0, 0])
        u_b = np.array([255, 255, 140])
        mask = cv2.inRange(mask, l_b, u_b)
        mask = cv2.equalizeHist(mask)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=6)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=9)
        return mask

    def create_advanced_mask(self, img):
        for k in range(6):
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            l_b = np.array([0, 0, 0])
            u_b = np.array([255, 255, 140])
            mask = cv2.inRange(mask, l_b, u_b)
            mask = cv2.equalizeHist(mask)
            mask = cv2.equalizeHist(mask)
            mask = cv2.medianBlur(mask, 15)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = self.fgbg.apply(mask)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3 + k, 3 + k)), iterations=7)
        return mask

    def show_frame(self, frame):
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)

    def stop_tracking(self):
        self.camera.StopGrabbing()
        cv2.destroyAllWindows()
        self.out.release()


if __name__ == "__main__":
    coclea_tracker = CocleaTracker()
    coclea_tracker.start_tracking()
    coclea_tracker.stop_tracking()
