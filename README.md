# Kumo Viewer 
Renderer for foggy object like smoke, cloud

![direct_beautiful](https://user-images.githubusercontent.com/42662735/145596076-375ee456-f26d-426d-8c26-768e3383f714.png)


![disney_cloud](https://user-images.githubusercontent.com/42662735/95776676-c074b580-0cff-11eb-9532-f2abf1d49d9a.png)


## Feature
- Volumetric Path Tracing
- Delta Tracking
- Distant Light with Ratio Tracking
- GPU Programming with CUDA
- using NanoVDB


## How to build

1. Download `wdas_cloud.vdb` from [here](https://disneyanimation.com/data-sets/).
2. Clone [nanovdb](https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb) and Build it.
3. Use `nanovdb_convert` in nanovdb to convert `wdas_cloud.vdb` to `wdas_cloud.nvdb`.
```
nanovdb_convert wdas_cloud.vdb wdas_cloud.nvdb
```
4. Move `wdas_cloud.nvdb` into `nvdbs` in this repository.
```
mv wdas_cloud.nvdb kumo_viewer/nvdbs
```
5. Build this repository
```
# example for windows
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build .
```
> I set CMAKE_CUDA_ARCHITECTURES=75 to tell cmake which CUDA architecture I use.
> Please refer [this page](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html).

6. Run!
