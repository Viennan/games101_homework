//
// Created by LEI XU on 5/16/19.
// Modified by Wang Yingnan on 8/23/22
//

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H
#include <cuda/std/limits>
#include "vector.cuh"
struct Ray{
    //Destination = origin + t*direction
    Vector3f origin;
    Vector3f direction, direction_inv;
    double t;//transportation time,
    double t_min, t_max;

    __host__ __device__ Ray(const Vector3f& ori, const Vector3f& dir, const double _t = 0.0): origin(ori), direction(dir),t(_t) {
        direction_inv = Vector3f(1./direction.x, 1./direction.y, 1./direction.z);
        t_min = 0.0;
        t_max = cuda::std::numeric_limits<double>::max();

    }

    __host__ __device__ Vector3f operator()(double t) const{return origin+direction*t;}
};
#endif //RAYTRACING_RAY_H