#ifndef API_HPP
#define API_HPP

#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "nanovdb/util/GridBuilder.h"
#include "nanovdb/util/IO.h"
#include "nanovdb/util/CudaDeviceBuffer.h"
#include "nanovdb/util/SampleFromVoxels.h"
#include "nanovdb/util/Ray.h"



#ifdef __CUDACC__
#define __twin__ __host__ __device__
#define __gpu__ __device__
#define __cpu__ __host__
#define __kernel__ __global__
#else
#define __twin__
#define __gpu__
#define __cpu__
#define __kernel__
#endif

#include "random.hpp"
#include "volume.hpp"
#include "render.hpp"


#endif