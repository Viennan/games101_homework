//
// Created by LEI XU on 5/13/19.
// Modified by Wang Yingnan on 8/23/22
//
#pragma once
#ifndef RAYTRACING_VECTOR_H
#define RAYTRACING_VECTOR_H

#include <iostream>
#include <cuda/std/cmath>
#include "global.cuh"

template<class T>
class Vector3 {
public:
    T x, y, z;
    __host__ __device__ Vector3() : x(0), y(0), z(0) {}
    __host__ __device__ Vector3(T xx) : x(xx), y(xx), z(xx) {}
    __host__ __device__ Vector3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}

    __host__ __device__ Vector3 operator * (float r) const { return Vector3(x * r, y * r, z * r); }

    __host__ __device__ Vector3 operator * (double r) const { return Vector3(T(x * r), T(y * r), T(z * r)); }

    __host__ __device__ Vector3 operator / (float r) const { return Vector3(x / r, y / r, z / r); }

    __host__ __device__ Vector3 operator / (double r) const { return Vector3(T(x / r), T(y / r), T(z / r)); }

    __host__ __device__ friend Vector3 operator * (float r, const Vector3& v) {return v * r;}
    __host__ __device__ friend Vector3 operator * (double r, const Vector3& v) {return v * r;}
    __host__ __device__ friend Vector3 operator / (double r, const Vector3& v) {return v / r;}

    __host__ __device__ T norm() {return cuda::std::sqrt(x * x + y * y + z * z);}
    __host__ __device__ Vector3 normalized() {

        T n = cuda::std::sqrt(x * x + y * y + z * z);
        return Vector3(x / n, y / n, z / n);
    }

    template<class U>
    __host__ __device__ Vector3 operator * (const Vector3<U> &v) const { return Vector3(x * v.x, y * v.y, z * v.z); }

    template<class U>
    __host__ __device__ Vector3 operator - (const Vector3<U> &v) const { return Vector3(x - v.x, y - v.y, z - v.z); }

    template<class U>
    __host__ __device__ Vector3 operator + (const Vector3<U> &v) const { return Vector3(x + v.x, y + v.y, z + v.z); }

    __host__ __device__ Vector3 operator - () const { return Vector3(-x, -y, -z); }

    template<class U>
    __host__ __device__ Vector3& operator += (const Vector3<U> &v) { x += v.x, y += v.y, z += v.z; return *this; }

    template<class U>
    __host__ __device__ Vector3& operator *= (const Vector3<U> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }

    friend std::ostream & operator << (std::ostream &os, const Vector3<T> &v)
    { return os << v.x << ", " << v.y << ", " << v.z; }

    __host__ __device__ T operator[](int index) const {
        return (&x)[index];
    }

    __host__ __device__ static Vector3 Min(const Vector3 &p1, const Vector3 &p2) {
        return Vector3(cuda_std_min(p1.x, p2.x), cuda_std_min(p1.y, p2.y), cuda_std_min(p1.z, p2.z));
    }

    __host__ __device__ static Vector3 Max(const Vector3 &p1, const Vector3 &p2) {
        return Vector3(cuda_std_max(p1.x, p2.x), cuda_std_max(p1.y, p2.y), cuda_std_max(p1.z, p2.z));
    }
};

using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

// class Vector3f {
// public:
//     float x, y, z;
//     __host__ __device__ Vector3f() : x(0), y(0), z(0) {}
//     __host__ __device__ Vector3f(float xx) : x(xx), y(xx), z(xx) {}
//     __host__ __device__ Vector3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
//     __host__ __device__ Vector3f operator * (const float &r) const { return Vector3f(x * r, y * r, z * r); }
//     __host__ __device__ Vector3f operator / (const float &r) const { return Vector3f(x / r, y / r, z / r); }

//     __host__ __device__ float norm() {return cuda::std::sqrt(x * x + y * y + z * z);}
//     __host__ __device__ Vector3f normalized() {

//         float n = cuda::std::sqrt(x * x + y * y + z * z);
//         return Vector3f(x / n, y / n, z / n);
//     }

//     __host__ __device__ Vector3f operator * (const Vector3f &v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
//     __host__ __device__ Vector3f operator - (const Vector3f &v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
//     __host__ __device__ Vector3f operator + (const Vector3f &v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
//     __host__ __device__ Vector3f operator - () const { return Vector3f(-x, -y, -z); }
//     __host__ __device__ Vector3f& operator += (const Vector3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
//     __host__ __device__ friend Vector3f operator * (const float &r, const Vector3f &v)
//     { return Vector3f(v.x * r, v.y * r, v.z * r); }

//     friend std::ostream & operator << (std::ostream &os, const Vector3f &v)
//     { return os << v.x << ", " << v.y << ", " << v.z; }

//     __host__ __device__ float  operator[](int index) const;

//     __host__ __device__ static Vector3f Min(const Vector3f &p1, const Vector3f &p2) {
//         return Vector3f(cuda_std_min(p1.x, p2.x), cuda_std_min(p1.y, p2.y), cuda_std_min(p1.z, p2.z));
//     }

//     __host__ __device__ static Vector3f Max(const Vector3f &p1, const Vector3f &p2) {
//         return Vector3f(cuda_std_max(p1.x, p2.x), cuda_std_max(p1.y, p2.y), cuda_std_max(p1.z, p2.z));
//     }
// };
// __host__ __device__ inline float Vector3f::operator[](int index) const {
//     return (&x)[index];
// }

class Vector2f
{
public:
    __host__ __device__ Vector2f() : x(0), y(0) {}
    __host__ __device__ Vector2f(float xx) : x(xx), y(xx) {}
    __host__ __device__ Vector2f(float xx, float yy) : x(xx), y(yy) {}
    __host__ __device__ Vector2f operator * (const float &r) const { return Vector2f(x * r, y * r); }
    __host__ __device__ Vector2f operator + (const Vector2f &v) const { return Vector2f(x + v.x, y + v.y); }
    float x, y;
};

__host__ __device__ inline Vector3f lerp(const Vector3f &a, const Vector3f& b, const float &t)
{ return a * (1 - t) + b * t; }

__host__ __device__ inline Vector3f normalize(const Vector3f &v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / cuda::std::sqrtf(mag2);
        return Vector3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

__host__ __device__ inline float dotProduct(const Vector3f &a, const Vector3f &b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__ inline Vector3f crossProduct(const Vector3f &a, const Vector3f &b)
{
    return Vector3f(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    );
}



#endif //RAYTRACING_VECTOR_H