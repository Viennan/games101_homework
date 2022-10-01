//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H

#include <atomic>
#include <vector>
#include <memory>
#include <ctime>
#include "object.cuh"
#include "ray.cuh"
#include "bounds.cuh"
#include "intersection.cuh"
#include "vector.cuh"

struct __align__(8) BVHBuildNode {
    union{
        Object** firstPrim;
        size_t right;
    };
    Bounds3 bounds;
    float area;
    uint32_t nPrimitives; // valid in leaf node
    int splitAxis;

    // BVHBuildNode Public Methods
    __host__ __device__ BVHBuildNode(): firstPrim(nullptr), bounds(), area(0.0f), nPrimitives(0), splitAxis(0)
    {}

    __host__ __device__ bool IsLeaf() const {
        return nPrimitives != 0;
    }
};

class BVHAccelCuda;

// BVHAccel Declarations
class BVHAccel {
public:
    friend class BVHAccelCuda;

    // BVHAccel Public Types
    enum class SplitMethod { NAIVE, SAH };

    // BVHAccel Public Methods
    BVHAccel(std::vector<Object*> p, uint32_t maxPrimsInNode = 1, uint32_t maxRecurDepth = MAX_BVH_DEPTH, SplitMethod splitMethod = SplitMethod::NAIVE);
    Bounds3 WorldBound() const;

    void Intersect(const Ray &ray, Intersection& isec) const;

    void Sample(Intersection &pos, float &pdf);

    BVHAccelCuda* moveToDevice(GpuMemAllocator& alloctor, Object** device) const;
    void mergeSubTree(uint32_t maxPrimsInSubTreeNode);
    void getParent(BVHAccel* p);

    // BVHAccel Private Data
    BVHBuildNode* root;
    uint32_t maxDepth;
    const SplitMethod splitMethod;
    std::vector<Object*> primitives;
    std::vector<BVHBuildNode> nodes; // It will be empty when root point to a middle node of other bvh tree.
    
private:
    uint32_t recursiveBuild(uint32_t beg, uint32_t End, uint32_t depth = 0, uint32_t maxRecurDepth = MAX_BVH_DEPTH, uint32_t maxPrimsInNode = 1);
};

class BVHAccelCuda {
public:    
    // the following members must be managed by cuda runtime
    BVHBuildNode *root;
    Object **objs;
    size_t maxDepth;
    size_t nNodes;

    __device__ void Intersect(const Ray &ray, Intersection& isec) const;

    __device__ void Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler) const;

private:
    BVHAccelCuda() = default;
};

#endif //RAYTRACING_BVH_H