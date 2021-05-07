cmake_minimum_required(VERSION 3.0)

project(evaluate_detect_3d_offline)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_BUILD_TYPE Debug)


add_executable(evaluate_detect_3d_offline evaluate_object_3d_offline.cpp)
