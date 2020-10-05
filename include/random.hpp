#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "api.hpp"


__kernel__ void random_init(int max_x, int max_y, curandState *rand_state) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   //Each thread gets same seed, a different sequence number, no offset
   curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


__gpu__ float rnd(curandState* rand_state)
{
    return curand_uniform(rand_state);
}

#endif