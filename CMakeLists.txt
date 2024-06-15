project(kumo_viewer)
cmake_minimum_required(VERSION 3.16)


set (CMAKE_CXX_STANDARD 11)

#ZLIB
# find_package(ZLIB REQUIRED)
add_subdirectory(ext/zlib-1.3.1)
add_definitions(-DNANOVDB_USE_ZIP)

#CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda Version: " ${CUDA_VERSION})
    set ( CMAKE_CUDA_STANDARD 11 )
	set ( CMAKE_CUDA_STANDARD_REQUIRED ON )
    set ( CMAKE_CUDA_FLAGS "--extended-lambda")
    enable_language(CUDA)
endif()

add_executable(kumo render.cu)

#extern
target_include_directories(kumo PUBLIC ext)
target_include_directories(kumo PUBLIC ext/zlib-1.3.1)

#include
target_include_directories(kumo PUBLIC include)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/path.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/path.hpp"
)

#Main
target_link_libraries(kumo PUBLIC zlibstatic)
