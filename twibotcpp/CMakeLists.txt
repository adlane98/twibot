cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(twibotcpp)

message("compiler ${CMAKE_CXX_COMPILER};${CMAKE_C_COMPILER}")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(twibotcpp ${SOURCE_FILES})
message("${OpenCV_LIBS}")

target_link_libraries(twibotcpp "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET twibotcpp PROPERTY CXX_STANDARD 14)


