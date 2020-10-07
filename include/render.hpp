#ifndef RENDER_HPP
#define RENDER_HPP

#include <string>
#include "api.hpp"


class RenderSetting
{
public:
    int width;
    int height;
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


__gpu__ void genFirstRay(int i, nanovdb::Vec3f& origin, nanovdb::Vec3f& dir)
{
    
}

__gpu__ nanovdb::Vec3f gamma(const nanovdb::Vec3f& color)
{
    nanovdb::Vec3f result;
    result[0] = fminf(powf(color[0], 1/2.2), 1.0);
    result[1] = fminf(powf(color[1], 1/2.2), 1.0);
    result[2] = fminf(powf(color[2], 1/2.2), 1.0);

    return result;
}

__kernel__ void renderKernel(const nanovdb::FloatGrid* grid, const nanovdb::Vec3f lightdir,
                            const float l_intensity, const float max_density,
                            int max_depth, const float sigma_s, const float sigma_a,
                            const float g,
                            float *fb, int max_x, int max_y, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;


    int pixel_index = j*max_x*3 + i*3;

    nanovdb::Vec3f cameraorigin, cameradir;
    genFirstRay(j * max_x + i, cameraorigin, cameradir);
    nanovdb::Ray<float> firstRay{cameraorigin, cameradir};

    nanovdb::Vec3f Color = RayTrace(grid, lightdir, l_intensity, rand_state, 
                                    max_density, max_depth, sigma_s, sigma_a, g, firstRay);

    Color = gamma(Color);

    fb[pixel_index + 0] = Color[0];
    fb[pixel_index + 1] = Color[1];
    fb[pixel_index + 2] = Color[2];
}

void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
            const RenderSetting& set_info);

#endif