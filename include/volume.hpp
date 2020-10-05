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

__gpu__ float distant_sample(curandState* rand_state, float sigma_t, float u)
{
    return -logf(1 - u)/sigma_t;
}

__gpu__ int sampleEvent(nanovdb::Vec3f& events, float u)
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


__gpu__ nanovdb::Vec3f RayTrace(const nanovdb::FloatGrid* grid,
                                curandState* rand_state,
                                const float max_density,
                                const int max_depth,
                                float sigma_s,
                                float sigma_a,
                                float g,
                                const nanovdb::Ray<float>& firstray)
{

    nanovdb::Vec3f lightdir{1,0,0};
    
    float max_t = (sigma_s + sigma_a) * max_density;

    auto acc = grid->getAccessor();
    nanovdb::TrilinearSampler<decltype(acc)> sampler(acc);
    nanovdb::Ray<float> wRay = firstray;
    auto wBbox = grid->worldBBox();

    nanovdb::Vec3f f{1,1,1};
    nanovdb::Vec3f pdfs{1,1,1};

    // start delta tracking
    for(int depth = 0; depth < max_depth; depth++)
    {
        //not intersect..
        if(wRay.clip(wBbox) == false) return {0,0,0};


        //intersect!!
        float t_near = wRay.t0();
        float t_far = wRay.t1();

        //distant sampling
        float t = t_near;
        float d_sampled = distant_sample(rand_state, max_t, rnd(rand_state));
        t += d_sampled;

         //transmit
        if(t >= t_far)
        {
            f = f * skycolor(wRay);
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
    }

    return f/pdfs;
}


#endif