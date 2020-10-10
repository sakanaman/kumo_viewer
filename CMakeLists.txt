project(kumo_viewer)
cmake_minimum_required(VERSION 3.16)


set (CMAKE_CXX_STANDARD 11)

#ZLIB
find_package(ZLIB REQUIRED)
add_definitions(-DNANOVDB_USE_ZIP)

#CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda Version: " ${CUDA_VERSION})
    set ( CMAKE_CUDA_STANDARD 11 )
	set ( CMAKE_CUDA_STANDARD_REQUIRED ON )
    set ( CMAKE_CUDA_FLAGS "--extended-lambda")
    enable_language(CUDA)
    enable_language(CUDA)
endif()

#extern
include_directories(ext)

#include
include_directories(include)

#zlib include
include_directories(${ZLIB_INCLUDE_DIR})


#Our Sources and Headers
# file(GLOB CPP_SOURCES LIST_DIRECTORIES false 
#     ${CMAKE_SOURCE_DIR}/src/*.cpp)
# message(STATUS ${CPP_SOURCES})
# file(GLOB CU_SOURCES LIST_DIRECTORIES false
#     ${CMAKE_SOURCE_DIR}/src/*.cu)
# message(STATUS ${CU_SOURCES})

# add_library(core STATIC ${CPP_SOURCES} ${CU_SOURCES})

#Main
add_executable(kumo render.cu)
target_link_libraries(kumo PRIVATE ZLIB::ZLIB)
