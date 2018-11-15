cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example-app example-app.cpp)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(nlohmann_json)

target_link_libraries(example-app "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)

set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
