cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#################################################
add_executable(mle #WIN32
    ${CMAKE_CURRENT_SOURCE_DIR}/src/NonLinearLSQ.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
#################################################