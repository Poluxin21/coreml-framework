cmake_minimum_required(VERSION 3.31)
project(raytracing)

set(CMAKE_CXX_STANDARD 20)

set(ONNXRUNTIME_DIR "D:/projects/cpp/raytracing/onnxruntime-win-x64-1.22.1")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(raytracing
        core/src/main.cpp
        core/include/ml_core.h
        core/src/ml_core.cpp
)

target_link_libraries(raytracing onnxruntime)

if (WIN32)
    target_link_libraries(raytracing ws2_32)
endif()
