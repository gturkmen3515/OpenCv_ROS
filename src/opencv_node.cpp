#include <ros/ros.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "opencv_node");
  ros::NodeHandle nh;

  // Your OpenCV code here

  ros::spin();
  return 0;
}

