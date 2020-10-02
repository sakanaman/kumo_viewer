#include "api.hpp"


int main()
{
    try
    {
        auto handle 
        = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>("../nvdbs/bunny_cloud.nvdb");
        auto* grid = handle.grid<float>();
        auto acc = grid->getAccessor();
        nanovdb::TrilinearSampler<decltype(acc)> samp(acc);
        std::cout << samp(nanovdb::Vec3<float>(40,10,0)) << std::endl;
        if(!grid)
            throw std::runtime_error("File did not contain a grid with value type float");
        if (handle.gridMetaData()->isFogVolume() == false) {
            throw std::runtime_error("Grid must be a fog volume");
        }
        std::cout << grid->gridName() << std::endl;
        std::cout << grid->worldBBox().dim()[0] << ", "<< grid->worldBBox().dim()[1] << ", "<< grid->worldBBox().dim()[2] <<  std::endl;
        // TODO: use setting file (eg. json, yaml...etc)
        RenderSetting set_info;
        set_info.height = 800;
        set_info.width = 1500;

        render(handle, set_info);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}