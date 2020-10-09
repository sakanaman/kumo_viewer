#include "api.hpp"


int main()
{
    try
    {
        //Load grid data from ~.nvdb
        auto handle 
        = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>("../nvdbs/bunny_cloud.nvdb");

        // Load Grid to GPU
        handle.deviceUpload();

        //get volume parametor from grid
        auto* h_grid = handle.grid<float>();
        auto acc = h_grid->getAccessor();
        auto ibbox = h_grid->indexBBox();
        float max_density = searchMaxDensity(ibbox, acc);

        // TODO: use setting file (eg. json, yaml...etc)
        RenderSetting set_info;
        set_info.height = 800;
        set_info.width = 1500;
        set_info.max_density = max_density;
        set_info.l_intensity = 10.f;
        set_info.sigma_a = 0.f;
        set_info.sigma_s = 1.5f;
        set_info.samples = 100;
        set_info.g = -0.1;
        set_info.max_depth = 100;

        //rendering standby...
        render(handle, set_info);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}