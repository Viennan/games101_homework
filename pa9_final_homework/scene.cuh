//
// Created by Göksu Güvendiren on 2019-05-14.
//
#ifndef RAYTRACING_SCENE_H
#define RAYTRACING_SCENE_H

#include <vector>
#include "vector.cuh"
#include "object.cuh"
#include "light.cuh"
#include "arealight.cuh"
#include "bvh.cuh"
#include "ray.cuh"

class SceneCuda;

class Scene
{
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 40;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    float RussianRoulette = 0.8;
    std::vector<Object* > objects;
    BVHAccel *bvh = nullptr;

    Scene(int w, int h) : width(w), height(h)
    {}

    ~Scene();

    void Add(Object *object) { objects.push_back(object); }
    const std::vector<Object*>& get_objects() const { return objects; }
    void intersect(const Ray& ray, Intersection& isec) const;
    
    void buildBVH(int maxDepth, int maxPrimInNodes);
    Vector3f castRay(const Ray &ray, int depth) const;
    void sampleLight(Intersection &pos, float &pdf) const;
    bool trace(const Ray &ray, const std::vector<Object*> &objects, float &tNear, uint32_t &index, Object **hitObject) const;
    SceneCuda* moveToDevice(GpuMemAllocator& alloctor) const;
};

class SceneCuda {
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 40;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    float RussianRoulette = 0.8;
    Object** objects; // managed
    BVHAccelCuda* bvh; // managed
    size_t nObjs;

    __device__ void intersect(const Ray& ray, Intersection& isec) const;
    __device__ void sampleLight(Intersection &pos, float &pdf, nvstd::function<float()> sampler) const;
    __device__ Vector3f castRay(const Ray &ray, nvstd::function<float()> sampler) const;

private:
    SceneCuda() = default;
};

#endif
