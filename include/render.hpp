#ifndef RENDER_HPP
#define RENDER_HPP

#include <string>
#include "api.hpp"

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



class RenderSetting
{
public:
    RenderSetting(){}
    int width;
    int height;
    nanovdb::Vec3f lightdir = {1,0,0};
    float l_intensity;
    float max_density;
    int max_depth;
    float sigma_s;
    float sigma_a;
    float g;
    int samples;
};

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


__gpu__ void genFirstRay(int i, int w, int h, nanovdb::Vec3f& origin, nanovdb::Vec3f& dir,
                         float wBBoxDimZ, const nanovdb::Vec3f& wBBoxCenter)
{
    int x = i % w;
    int y = i / w;

    const float fov = 45.f;
    const float u = (float(x) + 0.5f) / w;
    const float v = (float(y) + 0.5f) / h;

    const float aspect = w / float(h);
    const float px = (2.f * u - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f) * aspect;
    const float py = (2.f * v - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f);

    origin = wBBoxCenter + nanovdb::Vec3f(0, 0, wBBoxDimZ);
    dir = nanovdb::Vec3f(px, py, -1.f).normalize();
}

__gpu__ nanovdb::Vec3f gamma(const nanovdb::Vec3f& color)
{
    nanovdb::Vec3f result;
    result[0] = fmaxf(fminf(powf(color[0], 1/2.2), 1.0), 0.0);
    result[1] = fmaxf(fminf(powf(color[1], 1/2.2), 1.0), 0.0);
    result[2] = fmaxf(fminf(powf(color[2], 1/2.2), 1.0), 0.0);

    return result;
}

__kernel__ void renderKernel(const nanovdb::FloatGrid* grid, 
                            const RenderSetting& setting,
                            float *fb, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= setting.width) || (j >= setting.height)) return;

    auto wbbox = grid->worldBBox();
    auto dim = wbbox.dim();
    nanovdb::Vec3f center = nanovdb::Vec3f(wbbox.max() + wbbox.min()) * 0.5f;


    int pixel_index = j*setting.width*3 + i*3;

    // local randomstate
    curandState local_rand_state = rand_state[j * setting.width + i];

    //make first ray
    nanovdb::Vec3f cameraorigin, cameradir;
    genFirstRay(j * setting.width + i, setting.width, setting.height, cameraorigin, cameradir, 2.0 * dim[2], center);
    nanovdb::Ray<float> firstRay{cameraorigin, cameradir};

    // Let's Montecarlo
    nanovdb::Vec3f Color{};
    for(int i = 0; i < setting.samples; i++)
    {
        Color += RayTrace(grid, setting.lightdir, setting.l_intensity, &local_rand_state, 
                          setting.max_density, setting.max_depth, setting.sigma_s, setting.sigma_a, setting.g, firstRay);
    }
    Color /= float(setting.samples);

    // Gamma Process
    Color = gamma(Color);

    // write color buffer
    fb[pixel_index + 0] = Color[0];
    fb[pixel_index + 1] = Color[1];
    fb[pixel_index + 2] = Color[2];
}

void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
            const RenderSetting& set_info);

#endif