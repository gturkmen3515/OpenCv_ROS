cmake_minimum_required(VERSION 3.0.2)
project(OpenCv_ROS)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(opencv_node src/opencv_node.cpp)

target_link_libraries(opencv_node
  ${catkin_LIBRARIES}
)

install(TARGETS opencv_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)