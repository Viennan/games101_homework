//
// Created by Göksu Güvendiren on 2019-05-14.
//

#ifndef RAYTRACING_AREA_LIGHT_H
#define RAYTRACING_AREA_LIGHT_H

#include "vector.cuh"
#include "light.cuh"
#include "global.cuh"

// 40Bytes
class AreaLight : public Light
{
public:
    __host__ __device__ AreaLight(const Vector3f &p, const Vector3f &i) : Light(p, i)
    {
        normal = Vector3f(0, -1, 0);
        u = Vector3f(1, 0, 0);
        v = Vector3f(0, 0, 1);
        length = 100;
    }

    __host__ __device__ Vector3f SamplePoint(nvstd::function<float()> sampler) const
    {
        auto random_u = sampler();
        auto random_v = sampler();
        return position + random_u * u + random_v * v;
    }

    float length;
    Vector3f normal;
    Vector3f u;
    Vector3f v;
};

#endif