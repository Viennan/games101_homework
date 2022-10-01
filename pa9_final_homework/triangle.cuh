#ifndef RAYTRACING_TRIANGLE_H
#define RAYTRACING_TRIANGLE_H

#include <array>
#include <memory>
#include "vector.cuh"
#include "object.cuh"
#include "material.cuh"
#include "bvh.cuh"

__host__ __device__ inline bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1,
                          const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;
    Vector3f pvec = crossProduct(dir, edge2);
    float det = dotProduct(edge1, pvec);
    if (det == 0 || det < 0)
        return false;

    Vector3f tvec = orig - v0;
    u = dotProduct(tvec, pvec);
    if (u < 0 || u > det)
        return false;

    Vector3f qvec = crossProduct(tvec, edge1);
    v = dotProduct(dir, qvec);
    if (v < 0 || u + v > det)
        return false;

    float invDet = 1 / det;

    tnear = dotProduct(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}


// 128Bytes
__align__(8) class Triangle : public Object
{
public:
    Material* m;
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f t0, t1, t2; // texture coords
    Vector3f normal;
    float area;

    __host__ __device__ Triangle() = default;
    
    __host__ __device__ Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, Material* _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), m(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm()*0.5f;
    }

    __host__ __device__ bool intersect(const Ray& ray) override;
    __host__ __device__ bool intersect(const Ray& ray, float& tnear,
                   uint32_t& index) const override;
    __host__ __device__ void getIntersection(const Ray& ray, Intersection& isec) override;
    __host__ __device__ void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const override
    {
        N = normal;
        //        throw std::runtime_error("triangle::getSurfaceProperties not
        //        implemented.");
    }
    __host__ __device__ Vector3f evalDiffuseColor(const Vector2f&) const override;
    __host__ __device__ Bounds3 getBounds() override;

    __host__ __device__ void Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler){
        float x = cuda::std::sqrt(sampler()), y = sampler();
        pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
        pos.normal = this->normal;
        pdf = 1.0f / area;
    }
    __host__ __device__ float getArea(){
        return area;
    }
    __host__ __device__ bool hasEmit(){
        return m->hasEmission();
    }

    __host__ virtual Triangle* moveToDevice(GpuMemAllocator& alloctor) const override;
};

__host__ __device__ inline bool Triangle::intersect(const Ray& ray) { return true; }
__host__ __device__ inline bool Triangle::intersect(const Ray& ray, float& tnear, uint32_t& index) const
{
    return false;
}

__host__ __device__ inline Bounds3 Triangle::getBounds() { return Union(Bounds3(v0, v1), v2); }

__host__ __device__ inline void Triangle::getIntersection(const Ray& ray, Intersection& isec)
{
    if (dotProduct(ray.direction, normal) > 0)
        return;
    double u, v, t_tmp = 0;
    Vector3f pvec = crossProduct(ray.direction, e2);
    double det = dotProduct(e1, pvec);
    if (cuda::std::fabs(det) < EPSILON)
        return;

    double det_inv = 1. / det;
    Vector3f tvec = ray.origin - v0;
    u = dotProduct(tvec, pvec) * det_inv;
    if (u < 0 || u > 1)
        return;
    Vector3f qvec = crossProduct(tvec, e1);
    v = dotProduct(ray.direction, qvec) * det_inv;
    if (v < 0 || u + v > 1)
        return;
    t_tmp = dotProduct(e2, qvec) * det_inv;

    // TODO find ray triangle intersection
    if(t_tmp < 0){
        return;
    }

    if (t_tmp < isec.distance) {
        isec.distance = t_tmp;
        isec.happened = true;
        isec.m = m;
        isec.obj = this;
        isec.normal = normal;
        isec.coords = ray(t_tmp);
        isec.emit = m->getEmission();
    }
    return;
}

__host__ __device__ inline Vector3f Triangle::evalDiffuseColor(const Vector2f&) const
{
    return Vector3f(0.5, 0.5, 0.5);
}

class MeshTriangle : public Object
{
public:
    __host__ MeshTriangle(const std::string& filename, Material *mt = new Material());

    __host__ __device__ MeshTriangle():bounding_box(), numTriangles(0), triangles(nullptr), bvh(nullptr), bvh_cuda(nullptr), area(0.0f), m(nullptr) {};

    __host__ __device__ ~MeshTriangle();

    __host__ BVHAccel* buildBVH(int maxRecurDepth, uint32_t maxPrimsInNode) override;

    __host__ __device__ bool intersect(const Ray& ray) { return true; }

    __host__ __device__ bool intersect(const Ray& ray, float& tnear, uint32_t& index) const { return true;}

    __host__ __device__ Bounds3 getBounds() { return bounding_box; }

    __host__ __device__ void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const 
    {}

    __host__ __device__ Vector3f evalDiffuseColor(const Vector2f& st) const
    {
        float scale = 5;
        float pattern =
            (cuda::std::fmodf(st.x * scale, 1) > 0.5) ^ (cuda::std::fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031),
                    Vector3f(0.937, 0.937, 0.231), pattern);
    }

    __host__ __device__ void getIntersection(const Ray& ray, Intersection& isec) override {
#ifdef __CUDA_ARCH__
    bvh_cuda->Intersect(ray, isec);
#else
    bvh->Intersect(ray, isec);
#endif
    }
    
    __host__ __device__ void Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler){
#ifdef __CUDA_ARCH__
        bvh_cuda->Sample(pos, pdf, sampler);
#else
        bvh->Sample(pos, pdf);
#endif
        pos.emit = m->getEmission();
    }

    __host__ __device__ float getArea(){
        return area;
    }
    __host__ __device__ bool hasEmit(){
        return m->hasEmission();
    }

    __host__ MeshTriangle* moveToDevice(GpuMemAllocator& alloctor) const override;

    __host__ void Locate(float x, float y, float z, float zoom);

    Triangle* triangles;
    Material* m;
    BVHAccel* bvh;
    BVHAccelCuda* bvh_cuda;

    Bounds3 bounding_box;
    uint32_t numTriangles;
    float area; 
};

#endif
