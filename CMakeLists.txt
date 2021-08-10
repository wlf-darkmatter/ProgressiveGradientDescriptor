cmake_minimum_required(VERSION 3.20)
project(ProgressiveGradientDescriptor)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(include)
include_directories(source)
include_directories("/opt/homebrew/Cellar/opencv/4.5.3_1/include/opencv4")

add_executable(ProgressiveGradientDescriptor
        include/PGD.h
        source/PGD.cpp
        main.cpp)
