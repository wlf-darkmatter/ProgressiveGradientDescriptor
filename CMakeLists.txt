cmake_minimum_required(VERSION 3.20)
project(ProgressiveGradientDescriptor)
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(OpenCv_LIBS
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        )
# 设置opencv路径
set(OpenCv_path
        /opt/homebrew/Cellar/opencv/4.5.4_4
        )
include_directories(.)
include_directories(include)
include_directories(source)
include_directories(${OpenCv_path}/include/opencv4)

link_directories(
        ${OpenCv_path}/lib
)

add_executable(ProgressiveGradientDescriptor
        source/PGD.cpp
        include/PGD.h
        main.cpp)

target_link_libraries(ProgressiveGradientDescriptor ${OpenCv_LIBS}
        )
