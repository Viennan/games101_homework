//
// Created by LEI XU on 5/13/19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "object.cuh"
#include "vector.cuh"
#include "bounds.cuh"
#include "material.cuh"

class Sphere : public Object{
public:
    Vector3f center;
    float radius, radius2;
    float area;
    Material *m;
    Sphere(const Vector3f &c, const float &r, Material* mt = new Material()) : center(c), radius(r), radius2(r * r), m(mt), area(4 * M_PI *r *r) {}
    __host__ __device__ bool intersect(const Ray& ray) {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        float area = 4 * M_PI * radius2;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
    __host__ __device__ bool intersect(const Ray& ray, float &tnear, uint32_t &index) const
    {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!
        solveQuadratic(a, b, c, t0, t1)) return false;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        tnear = t0;

        return true;
    }

    __host__ __device__ void getIntersection(const Ray& ray, Intersection& isec) override {
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return;

        if (t0 < isec.distance) {
            isec.happened = true;
            isec.coords = Vector3f(ray.origin + ray.direction * t0);
            isec.normal = normalize(Vector3f(isec.coords - center));
            isec.m = this->m;
            isec.obj = this;
            isec.distance = t0;
            isec.emit = this->m->getEmission();
        }
    }

    __host__ __device__ void getSurfaceProperties(const Vector3f &P, const Vector3f &I, const uint32_t &index, const Vector2f &uv, Vector3f &N, Vector2f &st) const
    { N = normalize(P - center); }

    __host__ __device__ Vector3f evalDiffuseColor(const Vector2f &st)const {
        //return m->getColor();
    }
    __host__ __device__ Bounds3 getBounds(){
        return Bounds3(Vector3f(center.x-radius, center.y-radius, center.z-radius),
                       Vector3f(center.x+radius, center.y+radius, center.z+radius));
    }

    __host__ __device__ void Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler){
        float theta = 2.0 * M_PI * sampler(), phi = M_PI * sampler();
        Vector3f dir(cuda::std::cos(phi), cuda::std::sin(phi)*cuda::std::cos(theta), cuda::std::sin(phi)*cuda::std::sin(theta));
        pos.coords = center + radius * dir;
        pos.normal = dir;
        pos.emit = m->getEmission();
        pdf = 1.0f / area;
    }
    __host__ __device__ float getArea(){
        return area;
    }
    __host__ __device__ bool hasEmit(){
        return m->hasEmission();
    }
};

#endif //RAYTRACING_SPHERE_H