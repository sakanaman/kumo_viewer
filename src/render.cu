#include "api.hpp"
#include <iostream>
#include <fstream>
#include <time.h>



void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
    const RenderSetting& set_info)
{
auto* d_grid = handle.deviceGrid<float>();
auto* h_grid = handle.grid<float>();
auto acc = h_grid->getAccessor();
auto bbox = h_grid->indexBBox();


//get data from render setting
int nx = set_info.width;
int ny = set_info.height;
int tx = 8;
int ty = 8;
dim3 blocks(nx/tx + 1, ny/ty + 1);
dim3 threads(tx, ty);

//init random state per pixel
curandState *d_rand_state;
checkCudaErrors(cudaMalloc((void **)&d_rand_state, nx * ny * sizeof(curandState)));
random_init<<<blocks, threads>>>(nx, ny, d_rand_state);
checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaDeviceSynchronize());

std::cerr << "Rendering a " << nx << "x" << ny << " image ";
std::cerr << "in " << tx << "x" << ty << " blocks.\n";

int num_pixels = nx*ny;
size_t fb_size = 3*num_pixels*sizeof(float);

//malloc pixel buffer(for unified memory)
float* fb;
checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

//start time
clock_t start, stop;
start = clock();

//call kernel function
renderKernel<<<blocks, threads>>>(d_grid, set_info, fb, d_rand_state);
checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaDeviceSynchronize());

//finish time
stop = clock();
double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
std::cerr << "took " << timer_seconds << " seconds.\n";

//save ppm
SavePPM(fb, nx, ny);

//free several data
checkCudaErrors(cudaFree(fb));
}