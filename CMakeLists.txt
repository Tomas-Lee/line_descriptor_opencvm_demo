cmake_minimum_required(VERSION 3.5)
project(extract_match_line)

set(CMAKE_CXX_STANDARD 11)

#OpenCV
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

#glogging
find_package(Glog REQUIRED)

#line_descriptor_opencvm
include_directories(${PROJECT_SOURCE_DIR}/line_descriptor_opencvm/include)

add_executable(extract_match_line main.cpp)
target_link_libraries(extract_match_line
        ${OpenCV_LIBS}
        ${Glog_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/line_descriptor_opencvm/lib/liblinedesc.so
        )
