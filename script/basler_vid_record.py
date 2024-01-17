# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:19:48 2022

@author: atakan
"""
import pypylon.pylon as pylon
from imageio import get_writer

class CameraRecorder:
    def __init__(self, output_filename='output-filename_vid.avi', codec='libx264', quality=None, fps=120, time_to_record=60):
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.output_filename = output_filename
        self.codec = codec
        self.quality = quality
        self.fps = fps
        self.time_to_record = time_to_record

    def start_recording(self):
        self.cam.Open()
        print("Using device ", self.cam.GetDeviceInfo().GetModelName())
        writer = get_writer(
            self.output_filename,
            codec=self.codec,
            quality=self.quality,
            fps=self.fps,
            ffmpeg_params=[
                '-preset', 'fast',
                '-crf', '24'
            ]
        )
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        while self.cam.IsGrabbing():
            with self.cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException) as res:
                if res.GrabSucceeded():
                    img = res.Array
                    writer.append_data(img)
                    print(res.BlockID, end='\r')
                    res.Release()
                else:
                    print("Grab failed")

        print("Saving...", end=' ')
        self.cam.StopGrabbing()
        self.cam.Close()
        writer.close()
        print("Done")

if __name__ == "__main__":
    recorder = CameraRecorder()
    recorder.start_recording()
