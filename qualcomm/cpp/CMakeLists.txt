cmake_minimum_required(VERSION 3.14)
project(qualcomm_qnn_stub CXX)
add_executable(qnn_stub qnn_runtime_stub.cpp)
target_compile_features(qnn_stub PRIVATE cxx_std_17)
