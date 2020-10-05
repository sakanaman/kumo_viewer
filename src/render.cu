#include "api.hpp"
#include <iostream>
#include <fstream>
#include <time.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if(result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at" << 
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__gpu__ void genFirstRay(int i, nanovdb::Vec3f& origin, nanovdb::Vec3f& dir)
{
    
}

__kernel__ void renderKernel(float *fb, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

void SavePPM(float* fb, int nx, int ny)
{
    // Output FB as Image
    std::ofstream file("output.ppm");
    file << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            file << ir << " " << ig << " " << ib << "\n";
        }
    }
    file.close();
}

void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
            const RenderSetting& set_info)
{
    auto* d_grid = handle.deviceGrid();
    auto* h_grid = handle.grid<float>();
    auto acc = h_grid->getAccessor();
    auto bbox = h_grid->indexBBox();
    
    // check max density
    float max_density = searchMaxDensity(bbox, acc);

    //get data from render setting
    int nx = set_info.width;
    int ny = set_info.height;
    int tx = 8;
    int ty = 8;

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
    dim3 blocks(nx/tx + 1, ny/ty + 1);
    dim3 threads(tx, ty);
    renderKernel<<<blocks, threads>>>(fb, nx, ny);
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