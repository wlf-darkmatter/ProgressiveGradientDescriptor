cmake_minimum_required(VERSION 3.20)
project(ProgressiveGradientDescriptor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(OpenCv_LIBS
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs)

include_directories(.)
include_directories(include)
include_directories(source)
include_directories(/opt/homebrew/Cellar/opencv/4.5.3_1/include/opencv4)

link_directories(
        /opt/homebrew/Cellar/opencv/4.5.3_1/lib
)
add_executable(ProgressiveGradientDescriptor
        include/PGD.h
        source/PGD.cpp
        main.cpp)

target_link_libraries(ProgressiveGradientDescriptor ${OpenCv_LIBS}
        )