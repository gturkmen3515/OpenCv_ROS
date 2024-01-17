import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
from std_msgs.msg import Float64MultiArray
import rospy
from std_srvs.srv import Empty
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension
from math import atan2, cos, sin, sqrt, pi
from time import time
import pandas as pd
from pypylon import pylon
videoFilePath = "output_coclea.avi"

videoFile =cv2.VideoCapture(videoFilePath)

frame_width = int( videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( videoFile.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output.avi', fourcc, 1, (frame_height, frame_width))