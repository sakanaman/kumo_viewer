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

        // TODO: use setting file (eg. json, yaml...etc)
        RenderSetting set_info;
        set_info.height = 800;
        set_info.width = 1500;

        //rendering standby...
        render(handle, set_info);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}