cmake_minimum_required(VERSION 2.8)
project(showimage)
ADD_DEFINITIONS("-std=c++0x")
set(OpenCV_DIR, /usr/share/opencv/)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(showimage hough.cpp bird.cpp StegerLines.cpp HoughTransform.cpp
    GaussDeriv.cpp normal.cpp)
target_link_libraries(showimage opencv_core opencv_imgproc opencv_highgui)
