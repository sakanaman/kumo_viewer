#ifndef API_HPP
#define API_HPP

#include "nanovdb/util/GridBuilder.h"
#include "nanovdb/util/IO.h"
#include "nanovdb/util/CudaDeviceBuffer.h"
#include "nanovdb/util/SampleFromVoxels.h"


#ifdef __CUDACC__
#define __twin__ __host__ __device__
#define __gpu__ __device__
#define __cpu__ __host__
#else
#define __twin__
#define __gpu__
#define __cpu__
#endif

#include "render.hpp"


#endif