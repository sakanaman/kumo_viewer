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

void render(nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>& handle,
            const RenderSetting& set_info);

#endif