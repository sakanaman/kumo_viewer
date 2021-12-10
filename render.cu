#include "api.hpp"
#include <iostream>
#include <fstream>
#include <time.h>



void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
    const RenderSetting& setting)
{
auto* d_grid = handle.deviceGrid<float>();
if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");
auto* h_grid = handle.grid<float>();
auto acc = h_grid->getAccessor();
auto bbox = h_grid->indexBBox();


//get data from render setting
int nx = setting.width;
int ny = setting.height;
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

//render function
auto f = [setting] __gpu__ (int i, int j, float* fb, const nanovdb::FloatGrid* grid, curandState* rand_state)
{
    if((i >= setting.width) || (j >= setting.height)) return;

    auto wbbox = grid->worldBBox();
    auto dim = wbbox.dim();
    nanovdb::Vec3f center = nanovdb::Vec3f(wbbox.max() + wbbox.min()) * 0.5f;


    int pixel_index = j*setting.width*3 + i*3;

    // local randomstate
    curandState local_rand_state = rand_state[j * setting.width + i];

    //make first ray
    nanovdb::Vec3f cameraorigin, cameradir;
    genFirstRay(j * setting.width + i, setting.width, setting.height, cameraorigin, cameradir, 2.0 * dim[0], center);
    nanovdb::Ray<float> firstRay{cameraorigin, cameradir};

    // Let's Montecarlo
    nanovdb::Vec3f Color{};
    for(int i = 0; i < setting.samples; i++)
    {
        Color += RayTraceNEE(grid, setting.lightdir, setting.l_intensity, &local_rand_state, 
                          setting.max_density, setting.max_depth, setting.sigma_s, setting.sigma_a, setting.g, firstRay);
        // Color += RayTrace(grid, setting.lightdir, setting.l_intensity, &local_rand_state, 
        //                   setting.max_density, setting.max_depth, setting.sigma_s, setting.sigma_a, setting.g, firstRay);
    }
    Color /= float(setting.samples);

    // Gamma Process
    Color = gamma(Color);

    // write color buffer
    fb[pixel_index + 0] = Color[0];
    fb[pixel_index + 1] = Color[1];
    fb[pixel_index + 2] = Color[2];
};


//call kernel function
renderKernel<<<blocks, threads>>>(d_grid, fb, d_rand_state, f);
checkCudaErrors(cudaGetLastError());
checkCudaErrors(cudaDeviceSynchronize());

//finish time
stop = clock();
double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
std::cerr << "took " << timer_seconds << " seconds.\n";

//save ppm
SavePPM(fb, nx, ny);

//free several data
checkCudaErrors(cudaFree(d_rand_state));
checkCudaErrors(cudaFree(fb));
}



int main()
{
    try
    {
        //Load grid data from ~.nvdb
        auto handle 
        = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>("../../nvdbs/wdas_cloud.nvdb");

        // Load Grid to GPU
        handle.deviceUpload();

        //get volume parametor from grid
        auto* h_grid = handle.grid<float>();
        auto acc = h_grid->getAccessor();
        auto ibbox = h_grid->indexBBox();
        // float max_density = searchMaxDensity(ibbox, acc);
        float max_density = 1.0f;

        // TODO: use setting file (eg. json, yaml...etc)
        RenderSetting setting;
        setting.height = 512;
        setting.width = 700;
        setting.max_density = max_density;
        setting.l_intensity = 5.0f;
        setting.lightdir = {0.0, 0.0, 1.0};
        setting.sigma_a = 0.0f;
        setting.sigma_s = 0.09f;
        setting.samples = 1000;
        setting.g = -0.1;
        setting.max_depth = 100;

        //rendering standby...
        render(handle, setting);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}