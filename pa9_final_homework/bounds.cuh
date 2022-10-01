//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BOUNDS3_H
#define RAYTRACING_BOUNDS3_H
#include "ray.cuh"
#include "vector.cuh"
#include <cuda/std/limits>
#include <cuda/std/array>

// 24Bytes
class Bounds3
{
  public:
    Vector3f pMin, pMax; // two points to specify the bounding box
    __host__ __device__ Bounds3()
    {
        double minNum = cuda::std::numeric_limits<double>::lowest();
        double maxNum = cuda::std::numeric_limits<double>::max();
        pMax = Vector3f(minNum, minNum, minNum);
        pMin = Vector3f(maxNum, maxNum, maxNum);
    }
    __host__ __device__ Bounds3(const Vector3f p) : pMin(p), pMax(p) {}
    __host__ __device__ Bounds3(const Vector3f p1, const Vector3f p2)
    {
        pMin = Vector3f(cuda::std::fmin(p1.x, p2.x), cuda::std::fmin(p1.y, p2.y), cuda::std::fmin(p1.z, p2.z));
        pMax = Vector3f(cuda::std::fmax(p1.x, p2.x), cuda::std::fmax(p1.y, p2.y), cuda::std::fmax(p1.z, p2.z));
    }

    __host__ __device__ Vector3f Diagonal() const { return pMax - pMin; }
    __host__ __device__ int maxExtent() const
    {
        Vector3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    __host__ __device__ double SurfaceArea() const
    {
        Vector3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    __host__ __device__ Vector3f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    __host__ __device__ Bounds3 Intersect(const Bounds3& b)
    {
        return Bounds3(Vector3f(cuda::std::fmax(pMin.x, b.pMin.x), cuda::std::fmax(pMin.y, b.pMin.y),
                                cuda::std::fmax(pMin.z, b.pMin.z)),
                       Vector3f(cuda::std::fmin(pMax.x, b.pMax.x), cuda::std::fmin(pMax.y, b.pMax.y),
                                cuda::std::fmin(pMax.z, b.pMax.z)));
    }

    __host__ __device__ Vector3f Offset(const Vector3f& p) const
    {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    __host__ __device__ bool Overlaps(const Bounds3& b1, const Bounds3& b2)
    {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    __host__ __device__ bool Inside(const Vector3f& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
                p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    __host__ __device__ inline const Vector3f& operator[](int i) const
    {
        return (i == 0) ? pMin : pMax;
    }

    __host__ __device__ inline bool IntersectP(const Ray& ray, const Vector3f& invDir,
                           const cuda::std::array<int, 3>& dirisNeg) const;
};



__host__ __device__ inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir,
                                const cuda::std::array<int, 3>& dirIsNeg) const
{
    // invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
    // dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
    // TODO test if ray bound intersects
    auto tmin_v = (pMin - ray.origin) * invDir;
    auto tmax_v = (pMax - ray.origin) * invDir;
    if (dirIsNeg[0]) {
        cuda::std::swap(tmin_v.x, tmax_v.x);
    }
    if (dirIsNeg[1]) {
        cuda::std::swap(tmin_v.y, tmax_v.y);
    }
    if (dirIsNeg[2]) {
        cuda::std::swap(tmin_v.z, tmax_v.z);
    }

    auto tmin = cuda_std_max(tmin_v.x, cuda_std_max(tmin_v.y, tmin_v.z));
    auto tmax = cuda_std_min(tmax_v.x, cuda_std_min(tmax_v.y, tmax_v.z));

    return tmin < tmax + EPSILON && tmax > -EPSILON;
}

__host__ __device__ inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b1.pMin, b2.pMin);
    ret.pMax = Vector3f::Max(b1.pMax, b2.pMax);
    return ret;
}

__host__ __device__ inline Bounds3 Union(const Bounds3& b, const Vector3f& p)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b.pMin, p);
    ret.pMax = Vector3f::Max(b.pMax, p);
    return ret;
}

#endif // RAYTRACING_BOUNDS3_H