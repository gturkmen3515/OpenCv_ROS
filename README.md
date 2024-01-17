# OpenCv_ROS


## Object Tracker with OpenCV, ROS, and Basler Camera (Pylon Module)
This project implements an object tracker using OpenCV, ROS (Robot Operating System), and a Basler camera with the Pylon module. The code uses a Kalman filter for object tracking and publishes the processed data to a ROS topic.

## Dependencies
Make sure to have the following dependencies installed:

OpenCV: A powerful computer vision library. Install using sudo apt-get install libopencv-dev or follow the instructions on OpenCV website.
ROS (Melodic): A middleware framework designed for developing software for robots. Install using the instructions on the ROS website.
Basler Pylon Camera Software Suite: Software package for Basler cameras. Follow the instructions on the Basler website.
Package Dependencies
The ROS package has dependencies that need to be installed for proper functionality. 
* https://github.com/basler/pylon-ros-camera
* https://github.com/ros-perception/vision_opencv

## Pseudo Code


# Import necessary libraries
    import cv2
    import numpy as np
    from std_msgs.msg import Float64MultiArray
    import rospy
    from time import time
    from pypylon import pylon
    from imageio import get_writer

# Define the ObjectTracker class
    class ObjectTracker:
        def __init__(self):
            # Initialize the Kalman filter
            self.initialize_kalman_filter()
            # Initialize the Basler camera
            self.initialize_camera()
            # Initialize ROS node
            self.initialize_ros()

        def initialize_kalman_filter(self):
            # Set up parameters for the Kalman filter
            # (measurement matrix, transition matrix, process and measurement noise covariance, etc.)
            # Initial state is set as a random value
            # (You may want to adjust these parameters based on your specific application)

        def initialize_camera(self):
            # Set up the Basler camera
            # (Instantiating the camera, starting grabbing, setting image format converter, etc.)
            # (You need to install the Pylon library and configure the camera accordingly)

        def initialize_ros(self):
            # Initialize ROS node, set the publishing rate, and define the image topic
            # (You should replace 'talker' with your desired node name and '/img' with your topic name)
            # Set up ROS publisher for sending data
            # (You may want to adjust the queue_size based on your needs)

        def find_nearest_with_cost(self, c_time, spir_x, spir_y, pixel_x, pixel_y, prev_idx=None):
            # Function to find the nearest point with a cost calculation
            # (This function seems to be related to choosing a point for tracking)

        def run(self):
            # Main loop to run the object tracking
            # (Retrieve frames from the camera, process each frame, and publish the data to ROS)
            # (The loop also handles exiting the application when the user presses 'Esc')

        def convert_image(self, grab_result):
            # Convert the grabbed image result to a format usable by OpenCV
            # (This function seems to be using the Pylon library to convert the image)

        def process_frame(self, img, current_time):
            # Process each frame (image) obtained from the camera
            # (This is where your frame processing logic should be implemented)

        def publish_data(self, img, current_time):
            # Publish the processed data (image or other information) to a ROS topic
            # (This is where you can extract and publish relevant information)
            # (You need to populate the Float64MultiArray msg with your data)

        def get_data(self):
            # Function to extract data (not implemented in the provided code)
            # (You may need to implement this function based on your requirements)

        #Entry point of the script
        if __name__ == "__main__":
            # Instantiate the ObjectTracker class and run the tracking loop
            tracker = ObjectTracker()
            tracker.run()
