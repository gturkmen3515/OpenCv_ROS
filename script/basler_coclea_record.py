# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""

from pypylon import pylon
import cv2
from imageio import get_writer

class CameraRecorder:
    def __init__(self, output_filename='output-filename.avi', frame_rate=10, exposure_time=1500):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.camera.AcquisitionFrameRate.SetValue(frame_rate)
        self.camera.ExposureTime.SetValue(exposure_time)
        self.writer = get_writer(
            output_filename,
            codec='libx264',
            quality=None,
            ffmpeg_params=[
                '-preset', 'fast',
                '-crf', '24'
            ]
        )

    def start_recording(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
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
        return img

    def show_frame(self, frame):
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)
        self.writer.append_data(frame)

    def stop_recording(self):
        self.camera.StopGrabbing()
        cv2.destroyAllWindows()
        self.writer.close()

if __name__ == "__main__":
    recorder = CameraRecorder()
    recorder.start_recording()
    recorder.stop_recording()
