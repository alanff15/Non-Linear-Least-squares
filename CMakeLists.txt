cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenMP REQUIRED)

link_libraries(
    OpenMP::OpenMP_CXX
)

#################################################
add_executable(mle #WIN32
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
#################################################