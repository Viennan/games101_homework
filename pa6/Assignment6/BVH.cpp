#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);
    int msecs = diff * 1000 - (hrs * 3600000) - (mins * 60000) - secs *1000;

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs %i ms\n\n",
        hrs, mins, secs, msecs);
}

auto native_split(std::vector<Object*>& objects)
{
    Bounds3 centroidBounds;
    for (int i = 0; i < objects.size(); ++i)
        centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
    int dim = centroidBounds.maxExtent();
    switch (dim) {
    case 0:
        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
            return f1->getBounds().Centroid().x <
                    f2->getBounds().Centroid().x;
        });
        break;
    case 1:
        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
            return f1->getBounds().Centroid().y <
                    f2->getBounds().Centroid().y;
        });
        break;
    case 2:
        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
            return f1->getBounds().Centroid().z <
                    f2->getBounds().Centroid().z;
        });
        break;
    }
    return objects.begin() + (objects.size() / 2);
}

auto sah_split(std::vector<Object*>& objects)
{
    Bounds3 centroidBounds;
    for (int i = 0; i < objects.size(); ++i)
        centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
    int dim = centroidBounds.maxExtent();
    constexpr int nbuckets = 12;
    struct BucketInfo {
        Bounds3 box;
        int counts = 0;
    };
    BucketInfo buckets[nbuckets];
    for (const auto& obj_ptr: objects)
    {
        auto box = obj_ptr->getBounds();
        auto offset = centroidBounds.Offset(box.Centroid());
        int bucket_id = std::min<int>(nbuckets-1, nbuckets * offset[dim]);
        ++buckets[bucket_id].counts;
        buckets[bucket_id].box = Union(buckets[bucket_id].box, box);
    }
    float costs[nbuckets-1] = {0.0f};
    for (int i=0;i<nbuckets-1;++i)
    {
        Bounds3 l_box;
        int l_count = 0;
        for (int j=0;j<=i;++j)
        {
            l_count += buckets[j].counts;
            l_box = Union(l_box, buckets[j].box);
        }
        Bounds3 r_box;
        int r_count = objects.size() - l_count;
        for (int j=i+1;j<nbuckets;++j)
            r_box = Union(r_box, buckets[j].box);
        costs[i] = 0.125f + (l_count * l_box.SurfaceArea() + r_count * r_box.SurfaceArea()) / centroidBounds.SurfaceArea();
    }
    float min_cost = costs[0];
    int min_bucket = 0;
    for (int i=1;i<nbuckets-1;++i)
    {
        if (costs[i]<min_cost)
        {
            min_cost = costs[i];
            min_bucket = i;
        }
    }
    auto mid = std::partition(objects.begin(), objects.end(), 
    [=](Object* obj_ptr){
        auto box = obj_ptr->getBounds();
        auto offset = centroidBounds.Offset(box.Centroid());
        int bucket_id = std::min<int>(nbuckets-1, nbuckets * offset[dim]);
        return bucket_id <= min_bucket;
    });
    return mid;
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        auto middling = objects.begin();
        if (this->splitMethod == SplitMethod::SAH && objects.size() > 4)
            middling = sah_split(objects);
        else
            middling = native_split(objects);

        auto beginning = objects.begin();
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
    const auto& box = node->bounds;
    std::array<int, 3> dirIsNeg{ray.direction.x<0, ray.direction.y<0, ray.direction.z<0};
    if (!box.IntersectP(ray, ray.direction_inv, dirIsNeg))
        return Intersection();
    if (node->object)
    {
        return node->object->getIntersection(ray);
    }
    else
    {
        Intersection intersects[2] = {getIntersection(node->left, ray), getIntersection(node->right, ray)};
        if (intersects[0].happened && intersects[1].happened)
            return intersects[0].distance < intersects[1].distance ? intersects[0] : intersects[1];
        else if (intersects[0].happened)
            return intersects[0];
        else if (intersects[1].happened)
            return intersects[1];
        else
            return Intersection();
    }
}