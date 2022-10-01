//
// Created by LEI XU on 5/13/19.
// Modified by Wang Yingnan on 8/23/22
//
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "vector.cuh"
#include "global.cuh"
#include "bounds.cuh"
#include "ray.cuh"
#include "intersection.cuh"

#include <nvfunctional>

class BVHAccel;
class Object
{
public:
    __host__ __device__ Object() {}
    __host__ __device__ virtual ~Object() {}
    __host__ __device__ virtual bool intersect(const Ray& ray) = 0;
    __host__ __device__ virtual bool intersect(const Ray& ray, float &, uint32_t &) const = 0;
    __host__ __device__ virtual void getIntersection(const Ray& _ray, Intersection&) = 0;
    __host__ __device__ virtual void getSurfaceProperties(const Vector3f &, const Vector3f &, const uint32_t &, const Vector2f &, Vector3f &, Vector2f &) const = 0;
    __host__ __device__ virtual Vector3f evalDiffuseColor(const Vector2f &) const =0;
    __host__ __device__ virtual Bounds3 getBounds()=0;
    __host__ __device__ virtual float getArea()=0;
    __host__ __device__ virtual void Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler)=0;
    __host__ __device__ virtual bool hasEmit()=0;
    __host__ virtual Object* moveToDevice(GpuMemAllocator& alloctor) const {return nullptr;}
    __host__ virtual BVHAccel* buildBVH(int maxRecurDepth, uint32_t maxPrimsInNode) {return nullptr;}
};
#endif //RAYTRACING_OBJECT_H