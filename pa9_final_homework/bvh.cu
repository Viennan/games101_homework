#include <algorithm>
#include <cassert>
#include "bvh.cuh"
#include "global.cuh"
#include <unordered_map>
#include <queue>

BVHAccel::BVHAccel(std::vector<Object*> p, uint32_t maxPrimsInNode, uint32_t maxRecurDepth,
                   SplitMethod splitMethod)
    : maxDepth(0), splitMethod(splitMethod), primitives(std::move(p)), nodes(0)
{
    if (primitives.empty())
        return;

    recursiveBuild(0, primitives.size(), 0, maxRecurDepth, maxPrimsInNode);
    root = nodes.data(); 
}

uint32_t BVHAccel::recursiveBuild(uint32_t beg, uint32_t End, uint32_t depth, uint32_t maxRecurDepth, uint32_t maxPrimsInNode)
{
    maxDepth = std::max(maxDepth, depth + 1);
    this->nodes.emplace_back();
    size_t node_id = this->nodes.size() - 1;

    // Compute bounds of all primitives in BVH node
    uint32_t nPrimitives = End - beg;
    if (depth + 1 >= maxRecurDepth || nPrimitives <= maxPrimsInNode) {
        auto firstPrim = primitives.data()+beg;
        auto& node = nodes[node_id];
        node.firstPrim = firstPrim;
        node.nPrimitives = nPrimitives;
        double area = 0.0;
        Bounds3 bounds;
        for (uint32_t i=0;i<nPrimitives;++i) {
            firstPrim[i]->buildBVH(maxRecurDepth-(depth+1), maxPrimsInNode);
            area += firstPrim[i]->getArea();
            bounds = Union(bounds, firstPrim[i]->getBounds());
        }
        node.bounds = bounds;    
        node.area = area;
        return node_id;
    }
    else {
        Bounds3 centroidBounds;
        for (int i = beg; i < End; ++i)
            centroidBounds = Union(centroidBounds, primitives[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        auto begIter = primitives.begin() + beg;
        auto endIter = primitives.begin() + End;
        switch (dim) {
        case 0:
            std::sort(begIter, endIter, [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:
            std::sort(begIter, endIter, [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:
            std::sort(begIter, endIter, [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto mid = beg + nPrimitives / 2;
        auto left_id = recursiveBuild(beg, mid, depth +1, maxRecurDepth, maxPrimsInNode);
        auto right_id = recursiveBuild(mid, End, depth + 1, maxRecurDepth, maxPrimsInNode);
        auto& node = this->nodes[node_id];
        const auto& left = this->nodes[left_id];
        const auto& right = this->nodes[right_id];
        node.right = right_id;
        node.bounds = Union(left.bounds, right.bounds);
        node.area = left.area + right.area;
        node.splitAxis = dim;
    }

    return node_id;
}

__host__ __device__ static void bvh_intersect_impl(BVHBuildNode* root, const Ray& ray, Intersection& isec) {
    int depth = 0;
    uint32_t visit_stack[MAX_BVH_DEPTH];
    int curr = 0, visit_depth = 0;
    cuda::std::array<int, 3> dirIsNeg{ray.direction.x<0, ray.direction.y<0, ray.direction.z<0};
    while(true)
    {
        const auto& node = root[curr];
        const auto& bound = node.bounds;
        if (bound.IntersectP(ray, ray.direction_inv, dirIsNeg)) {
            if (node.IsLeaf()) {
                for (uint32_t i=0;i<node.nPrimitives;++i) {
                    node.firstPrim[i]->getIntersection(ray, isec);
                }
                if (visit_depth==0)
                    break;
                curr = visit_stack[--visit_depth];
            }
            else {
                if (dirIsNeg[node.splitAxis]) {
                    visit_stack[visit_depth++] = curr + 1;
                    curr = node.right;
                }
                else {
                    visit_stack[visit_depth++] = node.right;
                    ++curr;
                }
            }
        }
        else {
            if (visit_depth == 0) 
                break;
            curr = visit_stack[--visit_depth];
        } 
    }
    return;
}

void BVHAccel::Intersect(const Ray& ray, Intersection& isec) const
{
    // TODO Traverse the BVH to find intersection
    bvh_intersect_impl(this->root, ray, isec);
}

__host__ __device__ static void getSample(BVHBuildNode* root, float p, Intersection &pos, nvstd::function<float()> sampler, float &pdf){
    uint32_t offset = 0;
    double area = 0.0;
    for (int i=0;i<MAX_BVH_DEPTH;++i) {
        const auto& node = root[offset];
        if (root[offset].IsLeaf()) {
            auto primitive_ptr = node.firstPrim;
            for (uint32_t i=0;i<node.nPrimitives;++i, ++primitive_ptr) {
                auto primitive = *primitive_ptr;
                auto p_area = primitive->getArea();
                area += p_area;
                if (area >= p) {
                    primitive->Sample(pos, pdf, sampler);
                    pdf *= p_area;
                    break;
                }
            }
            return;
        } 
        else {
            if (p < area + root[offset+1].area) {
                ++offset;
            } else {
                area += node.area;
                offset = node.right;
            }
        }
    }
}

void BVHAccel::Sample(Intersection &pos, float &pdf){
    float p = get_random_float() * root->area;
    getSample(root, p, pos, get_random_float, pdf);
    pdf /= root->area;
}

BVHAccelCuda* BVHAccel::moveToDevice(GpuMemAllocator& alloctor, Object** device) const {
    BVHAccelCuda* bvh_cuda;
    size_t nNodes = nodes.size();
    bvh_cuda = alloctor.allocManaged(&bvh_cuda, sizeof(BVHAccelCuda));
    bvh_cuda->root = alloctor.allocManaged(&bvh_cuda->root, sizeof(BVHBuildNode) * nNodes);
    bvh_cuda->maxDepth = this->maxDepth;
    bvh_cuda->objs = device;
    bvh_cuda->nNodes = nNodes;
    for (size_t i=0,offset=0;i<nNodes;++i) {
        bvh_cuda->root[i] = this->root[i];
        if (this->root[i].IsLeaf()) {
            bvh_cuda->root[i].firstPrim = device + offset;
            offset += this->root[i].nPrimitives;
        }
    }
    return bvh_cuda;
}

 __device__ void BVHAccelCuda::Intersect(const Ray& ray, Intersection& isec) const {
    // TODO Traverse the BVH to find intersection
    bvh_intersect_impl(this->root, ray, isec);
 }

__device__ void BVHAccelCuda::Sample(Intersection &pos, float &pdf, nvstd::function<float()> sampler) const {
    float p = sampler() * root->area;
    getSample(root, p, pos, sampler, pdf);
    pdf /= root->area;
}
