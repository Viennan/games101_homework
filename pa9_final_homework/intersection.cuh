//
// Created by LEI XU on 5/16/19.
// Modified by Wang Yingnan on 8/23/22
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "vector.cuh"
#include "material.cuh"
class Object;
class Sphere;

struct __align__(8) Intersection
{
    __host__ __device__ Intersection(): happened(false), coords(), normal(), distance(cuda::std::numeric_limits<double>::max()), obj(nullptr), m(nullptr)
    {};
    Object* obj;
    Material* m;
    double distance;
    Vector3f coords;
    Vector3f tcoords;
    Vector3f normal;
    Vector3f emit;
    bool happened;
};
#endif //RAYTRACING_INTERSECTION_H