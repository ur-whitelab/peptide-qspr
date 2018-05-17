cmake_minimum_required(VERSION 2.8.12)
project(gibbs)
# Set source directory
set(SOURCE_DIR "peptideqspr/gibbs")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES "${SOURCE_DIR}/gibbs_py.cpp")

add_subdirectory("${SOURCE_DIR}")
