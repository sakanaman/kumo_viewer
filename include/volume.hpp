#ifndef VOLUME_HPP
#define VOLUME_HPP

#include "api.hpp"

//////////////////////////////////////////////////
//                HELPER FUNCTIONS              //
//////////////////////////////////////////////////
__gpu__ void branchlessONB(const nanovdb::Vec3f &n, nanovdb::Vec3f &b1, nanovdb::Vec3f &b2)
{
    float sign = copysignf(1.0f, n[2]);
    const float a = -1.0f / (sign + n[2]);
    const float b = n[0] * n[1] * a;
    b1 = nanovdb::Vec3f(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
    b2 = nanovdb::Vec3f(b, sign + n[1] * n[1] * a, -n[1]);
}

__gpu__ nanovdb::Vec3f local2world(const nanovdb::Vec3f& local, const nanovdb::Vec3f& x, const nanovdb::Vec3f& y, const nanovdb::Vec3f& z)
{
    return x * local[0] + y * local[1] + z * local[2];
}

__gpu__ nanovdb::Vec3f skycolor(const nanovdb::Ray<float>& r) {
    nanovdb::Vec3f unit_direction = r.dir();
    float t = 0.5f*(unit_direction[1] + 1.0f);
    return (1.0f-t)*nanovdb::Vec3f(1.0, 1.0, 1.0) + t*nanovdb::Vec3f(0.5, 0.7, 1.0);
}

// We don`t use this function when we render "wdas_cloud.nvdb" because it's so heavy.
// So, please set max_density = 1.0 when you render wdas_cloud.
__cpu__ float searchMaxDensity(const nanovdb::CoordBBox& bbox,
                               const nanovdb::ReadAccessor<nanovdb::NanoRoot<float>>& acc)
{
    // check max density
    float max_density = -100;
    for(nanovdb::CoordBBox::Iterator iter = bbox.begin(); iter; ++iter)
    {
        auto coord = *iter;
        float candidate = acc.getValue(coord);
        if(max_density < candidate)
        {
            max_density = candidate;
        }
    }
    return max_density;
}



//////////////////////////////////////////////////
//            CORE LOGIC FUNCTIONS              //
//////////////////////////////////////////////////
__gpu__ float henyey_greenstein_phase(float theta, float g)
{
    const float pi = 3.14159265358979323846f;
    float denomi = 1 + g*g - 2*g*cosf(theta);
    return 1/(4 * pi) * (1 - g*g) / powf(denomi, 3.0f/2.0f);
}

__gpu__ nanovdb::Vec3f henyey_greenstein_sample(float g, float u, float v)
{
    const float pi = 3.14159265358979323846f;
    float s = 2*u - 1;
    float T = (1 - g*g) / (1 + g * s);
    float cosTheta = 1.0/(2.0 * g) * (1 + g*g - powf(T, 2.0f));
    float sinTheta = sqrtf(1 - cosTheta*cosTheta);
    float phi = 2 * pi * v;

    //Note: This renderer assumes that Y-up.
    //      So This return has to be interpreted as {x, z, y} 
    return {sinTheta * cosf(phi),
            sinTheta * sinf(phi),
            cosTheta};
}

__gpu__ float distant_sample(float sigma_t, float u)
{
    return -logf(1 - u)/sigma_t;
}

__gpu__ int sampleEvent(const nanovdb::Vec3f& events, float u)
{
    // 0-->absorp, 1-->scatter, 2-->null scatter
    nanovdb::Vec3f cdf = nanovdb::Vec3f{events[0], 
                                        events[0] + events[1], 
                                        events[0] + events[1] + events[2]}
                                      /(events[0] + events[1] + events[2]);

    if(u < cdf[0])
    {
        return 0;
    }
    else if(u < cdf[1])
    {
        return 1;
    }
    return 2;
}

__gpu__ nanovdb::Vec3f SunLightNEE(const nanovdb::Vec3f& shadowRayorigin,
                                   const nanovdb::Vec3f& shadowRaydir,
                                   curandState* rand_state,
                                   const nanovdb::FloatGrid* grid,
                                   const float l_intensity,
                                   const float max_t,
                                   const float sigma_s,
                                   const float sigma_a)
{
    auto acc = grid->getAccessor();
    auto wBbox = grid->worldBBox();
    nanovdb::TrilinearSampler<decltype(acc)> sampler(acc);

    nanovdb::Ray<float> shadowRay{shadowRayorigin, shadowRaydir};
    // check intersect bbox and ray
    shadowRay.clip(wBbox); 

    //Ratio Tracing
    nanovdb::Vec3f throughput{1.0, 1.0, 1.0};
    float t = shadowRay.t0();
    float t_far = shadowRay.t1();
    while(true)
    {
        //distance sampling
        t += distant_sample(max_t, rnd(rand_state));

        //sampled distance is out of volume --> break
        if(t >= t_far)
        {
            break;
        }

        // calculate several parametor in now position
        float density = sampler(grid->worldToIndexF(shadowRay(t)));
        float absorp_weight = sigma_a * density;
        float scatter_weight = sigma_s * density;
        float null_weight = max_t - absorp_weight - scatter_weight;
        nanovdb::Vec3f events{absorp_weight, scatter_weight, null_weight};

        //sample event
        int e = sampleEvent(events, rnd(rand_state));

        if(e == 1 || e == 2)
        {
            break;
        }
        else
        {
            throughput *= null_weight/max_t;
        }
    }
    return throughput * l_intensity;
}


__gpu__ nanovdb::Vec3f RayTrace(const nanovdb::FloatGrid* grid,
                                const nanovdb::Vec3f& lightdir,
                                const float l_intensity,
                                curandState* rand_state,
                                const float max_density,
                                const int max_depth,
                                float sigma_s,
                                float sigma_a,
                                float g,
                                const nanovdb::Ray<float>& firstray)
{
    
    float max_t = (sigma_s + sigma_a) * max_density;

    auto acc = grid->getAccessor();
    nanovdb::TrilinearSampler<decltype(acc)> sampler(acc);
    nanovdb::Ray<float> wRay = firstray;
    auto wBbox = grid->worldBBox();

    nanovdb::Vec3f f{1,1,1};
    nanovdb::Vec3f pdfs{1,1,1};
    nanovdb::Vec3f L{};

    // start delta tracking
    for(int depth = 0; depth < max_depth; depth++)
    {
        //not intersect..
        if(wRay.clip(wBbox) == false)
        {
            L = skycolor(wRay);
            break;
        }


        //intersect!!
        float t_near = wRay.t0();
        float t_far = wRay.t1();

        //distant sampling
        float t = t_near;
        float d_sampled = distant_sample(max_t, rnd(rand_state));
        t += d_sampled;

         //transmit
        if(t >= t_far)
        {
            f = f * skycolor(wRay);
            L += f/pdfs;
            break;
        }

        //sample density
        auto pos = wRay(t);
        auto ipos = grid->worldToIndexF(pos);
        float density = sampler(ipos);

        //calculate several parametor
        float absorp_weight = sigma_a * density;
        float scatter_weight = sigma_s * density;
        float null_weight = max_t - absorp_weight - scatter_weight;
        nanovdb::Vec3f events{absorp_weight, scatter_weight, null_weight};

        //Sample Event
        // 0-->absorp, 1-->scatter, 2-->null scatter
        int e = sampleEvent(events, rnd(rand_state));


        if(e == 0)//absorp
        {
            //Todo: correspond to emission
            break;
        }
        else if (e == 1)//scatter
        {
            //NEE for sunlight
            float theta = acosf(lightdir.dot(wRay.dir()));
            float nee_phase = henyey_greenstein_phase(theta, g);
            L += f/pdfs * nee_phase * SunLightNEE(pos, lightdir, rand_state, grid, l_intensity, max_t, sigma_s, sigma_a);  

            //make next scatter Ray
            //   localize
            nanovdb::Vec3f b1;
            nanovdb::Vec3f b2;
            branchlessONB(wRay.dir(), b1, b2);
            //   sample scatter dir
            nanovdb::Vec3f local_scatterdir =  henyey_greenstein_sample(g, rnd(rand_state), rnd(rand_state));
            //   reset local ray to world ray
            nanovdb::Vec3f scatterdir = local2world(local_scatterdir, b1, b2, wRay.dir());
            //   reset ray
            wRay = nanovdb::Ray<float>{pos, scatterdir};

            //NOTE: you don't have to calculate throuput here!!
        }
        else // null scatter
        {
            // renew ray
            wRay = nanovdb::Ray<float>{pos, wRay.dir()};
            continue;
        }
        
    }

    return L;
}


#endif