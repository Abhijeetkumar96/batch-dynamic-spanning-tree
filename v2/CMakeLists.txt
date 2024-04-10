cmake_minimum_required(VERSION 3.18)
project(dynamic_spanning_tree LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

include_directories(include)

file(GLOB_RECURSE CUDA_SRC
    "src/common/*.cu"
    "src/dynamic_spanning_tree/*.cu"
    "src/PR-RST/*.cu"
    "src/hash_table/*.cu"
    "src/super_graph/*.cu"
    "src/cuda_bfs/*.cu"
    "src/connected_components/*.cu"
    "src/main.cu"
)

file(GLOB_RECURSE CPP_SRC
    "src/serial_rst/*.cpp"
)

add_executable(${PROJECT_NAME}
    ${CUDA_SRC}
    ${CPP_SRC}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 80)
