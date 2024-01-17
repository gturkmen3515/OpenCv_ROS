# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:34:47 2022

@author: atakan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:34:47 2022

@author: atakan
"""

from pypylon import pylon
import cv2

class BaslerCamera:
    def __init__(self, output_file='video/output_basler.avi', output_width=600, output_height=480, fps=20.0):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (output_width, output_height))
    
    def start_capture(self):
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
        frame = image.GetArray()
        img = cv2.flip(frame, 0)
        img = cv2.flip(img, 1)
        scale_percent = 60
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        img = cv2.resize(img, dsize)
        return img
    
    def show_frame(self, frame):
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)
        self.out.write(frame)
    
    def stop_capture(self):
        self.camera.StopGrabbing()
        cv2.destroyAllWindows()
        self.out.release()

if __name__ == "__main__":
    basler_camera = BaslerCamera()
    basler_camera.start_capture()
    basler_camera.stop_capture()

