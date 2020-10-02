project(kumo_viewer)
cmake_minimum_required(VERSION 3.16)


#ZLIB
find_package(ZLIB REQUIRED)
add_definitions(-DNANOVDB_USE_ZIP)

#CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda Version: " ${CUDA_VERSION})
    enable_language(CUDA)
endif()

#extern(for header only)
include_directories(ext)

#include
include_directories(include)

#zlib include ???
include_directories(${ZLIB_INCLUDE_DIR})


#Our Sources and Headers
file(GLOB CPP_SOURCES LIST_DIRECTORIES false 
    ${CMAKE_SOURCE_DIR}/src/*.cpp)
message(STATUS ${CPP_SOURCES})
file(GLOB CU_SOURCES LIST_DIRECTORIES false
    ${CMAKE_SOURCE_DIR}/src/*.cu)
message(STATUS ${CU_SOURCES})

add_library(core STATIC ${CPP_SOURCES} ${CU_SOURCES})

#Main
add_executable(kumo main.cpp)
target_link_libraries(kumo PRIVATE ZLIB::ZLIB core)
