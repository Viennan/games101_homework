//
// Created by Göksu Güvendiren on 2019-05-14.
//
#ifndef RAYTRACING_LIGHT_H
#define RAYTRACING_LIGHT_H

#include "vector.cuh"

__align__(8) class Light
{
public:
    __host__ __device__ Light(const Vector3f &p, const Vector3f &i) : position(p), intensity(i) {}
    __host__ __device__ virtual ~Light() {};
    Vector3f position;
    Vector3f intensity;
};

#endif
