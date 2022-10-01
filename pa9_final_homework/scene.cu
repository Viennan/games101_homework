//
// Created by Göksu Güvendiren on 2019-05-14.
// Modified by wangyingnan on 12/9/22
//

#include "scene.cuh"
#include <queue>

#define MAX_REFLECT_TIMES 256

#define PDF_EPSILON 0.000005f
const Vector3f ZERO3F = Vector3f();

Scene::~Scene() {
    if (bvh) {
        delete bvh;
        bvh = nullptr;
    }
}

void Scene::buildBVH(int maxDepth, int maxPrimInNodes) {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, maxPrimInNodes, maxDepth, BVHAccel::SplitMethod::NAIVE);
}

void Scene::intersect(const Ray &ray, Intersection& isec) const
{
    this->bvh->Intersect(ray, isec);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    float emit_area_acc = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            auto area = objects[k]->getArea();
            emit_area_acc += area;
            if (p <= emit_area_acc){
                objects[k]->Sample(pos, pdf, get_random_float);
                pdf *= area / emit_area_sum;
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject) const
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = cuda::std::numeric_limits<float>::max();
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }

    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here

    /***************************  Recursive implementation  **************************/
    // Intersection intersec;
    // this->intersect(ray, intersec);
    // if (!intersec.happened) {
    //     return Vector3f();
    // }
    // if (intersec.m->hasEmission()) {
    //     return depth == 0 ? dotProduct(intersec.normal, -ray.direction) * intersec.emit: Vector3f();
    // }
    
    // Vector3f l_dir;
    // Intersection l_pos;
    // float pdf = 1.0f;
    // this->sampleLight(l_pos, pdf);
    // auto obj2l = l_pos.coords - intersec.coords;
    // auto l_o_dir = obj2l.normalized();
    // Ray l_o(intersec.coords + EPSILON * intersec.normal, l_o_dir);
    // auto r = obj2l.norm();
    // auto r_pow = obj2l.x * obj2l.x + obj2l.y * obj2l.y + obj2l.z * obj2l.z;
    // Intersection mid_intersec;
    // this->intersect(l_o, mid_intersec);
    // if (pdf > PDF_EPSILON && mid_intersec.happened && r <= mid_intersec.distance + 3 * EPSILON) {
    //     auto cos_theta = dotProduct(intersec.normal, l_o_dir);
    //     auto cos_theta_x = dotProduct(l_pos.normal, -l_o_dir);
    //     auto f_r = intersec.m->eval(ray.direction, l_o_dir, intersec.normal);
    //     l_dir = f_r * l_pos.emit * cos_theta * cos_theta_x / (r_pow * pdf);
    // }

    // Vector3f l_indir;
    // if (get_random_float() < this->RussianRoulette) {
    //     auto ray_o_dir = normalize(intersec.m->sample(ray.direction, intersec.normal, get_random_float));
    //     auto ray_o = Ray(intersec.coords + EPSILON * ray_o_dir, ray_o_dir);
    //     auto pdf_indir = intersec.m->pdf(ray.direction, ray_o_dir, intersec.normal);
    //     if (pdf_indir > PDF_EPSILON) {
    //         auto f_r = intersec.m->eval(ray.direction, ray_o_dir, intersec.normal);
    //         auto cos_theta = dotProduct(intersec.normal, ray_o_dir);
    //         l_indir = f_r * this->castRay(ray_o, depth + 1) * cos_theta / pdf_indir /this->RussianRoulette;
    //     }
    // }

    // return Vector3f::Max(l_dir, ZERO3F) + Vector3f::Max(l_indir, ZERO3F);    

    /***************************  Iterative implementation  **************************/
    Vector3d l;
    Vector3d coef(1.0);
    Ray r_o = ray;
    for (int i=0;i<MAX_REFLECT_TIMES;++i) {
        Intersection intersec;
        this->intersect(r_o, intersec);
        if (!intersec.happened) {
            break;
        }
        if (intersec.m->hasEmission()) {
            l += coef * (i == 0 ? dotProduct(intersec.normal, -ray.direction) * intersec.emit: Vector3f());
            break;
        }
        
        // compute direct light
        Vector3f l_dir;
        Intersection l_pos;
        float pdf = 1.0f;
        this->sampleLight(l_pos, pdf);
        auto obj2l = l_pos.coords - intersec.coords;
        auto l_o_dir = obj2l.normalized();
        Ray l_o(intersec.coords + EPSILON * intersec.normal, l_o_dir);
        auto r = obj2l.norm();
        auto r_pow = obj2l.x * obj2l.x + obj2l.y * obj2l.y + obj2l.z * obj2l.z;
        Intersection mid_intersec;
        this->intersect(l_o, mid_intersec);
        if (pdf > PDF_EPSILON && mid_intersec.happened && r <= mid_intersec.distance + 3 * EPSILON) {
            auto cos_theta = dotProduct(intersec.normal, l_o_dir);
            auto cos_theta_x = dotProduct(l_pos.normal, -l_o_dir);
            auto f_r = intersec.m->eval(r_o.direction, l_o_dir, intersec.normal);
            l_dir = f_r * l_pos.emit * cos_theta * cos_theta_x / (r_pow * pdf);
        }
        l += Vector3d::Max(coef * l_dir, Vector3d());

        if (get_random_float() > this->RussianRoulette)
            break;

        auto ray_o_dir = normalize(intersec.m->sample(ray.direction, intersec.normal, get_random_float));
        r_o = Ray(intersec.coords + EPSILON * ray_o_dir, ray_o_dir);
        auto pdf_indir = intersec.m->pdf(ray.direction, ray_o_dir, intersec.normal);
        if (pdf_indir < PDF_EPSILON) {
            break;
        }
        auto f_r = intersec.m->eval(ray.direction, ray_o_dir, intersec.normal);
        auto cos_theta = dotProduct(intersec.normal, ray_o_dir);
        coef *= f_r * cos_theta / pdf_indir / this->RussianRoulette;
    }
    return Vector3f(l.x, l.y, l.z);
}

SceneCuda* Scene::moveToDevice(GpuMemAllocator& alloctor) const {
    SceneCuda* s_cuda;
    s_cuda = alloctor.allocManaged(&s_cuda, sizeof(SceneCuda));

    Object** objs_cuda;
    auto num = this->objects.size();
    objs_cuda = alloctor.allocManaged(&objs_cuda, sizeof(Object*) * num);
    for(size_t i=0;i<num;++i) {
        objs_cuda[i] = this->bvh->primitives[i]->moveToDevice(alloctor);
    }
    s_cuda->objects = objs_cuda;
    s_cuda->bvh = bvh->moveToDevice(alloctor, objs_cuda);
    s_cuda->width = this->width;
    s_cuda->height = this->height;
    s_cuda->fov = this->fov;
    s_cuda->backgroundColor = this->backgroundColor;
    s_cuda->RussianRoulette = this->RussianRoulette;
    s_cuda->nObjs = num;

    return s_cuda;
}

__device__ void SceneCuda::intersect(const Ray& ray, Intersection& isec) const {
    this->bvh->Intersect(ray, isec);
}

__device__ void SceneCuda::sampleLight(Intersection &pos, float &pdf, nvstd::function<float()> sampler) const {
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < nObjs; ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = sampler() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < nObjs; ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf, sampler);
                break;
            }
        }
    }
}

__device__ Vector3f SceneCuda::castRay(const Ray &ray, nvstd::function<float()> sampler) const {
    Vector3d l;
    Vector3d coef(1.0);
    Ray r_o = ray;
    for (int i=0;i<MAX_REFLECT_TIMES;++i) {
        Intersection intersec;
        this->intersect(r_o, intersec);
        if (!intersec.happened) {
            break;
        }
        if (intersec.m->hasEmission()) {
            l += coef * (i == 0 ? dotProduct(intersec.normal, -ray.direction) * intersec.emit: Vector3f());
            break;
        }

        // compute direct light
        Vector3f l_dir;
        Intersection l_pos;
        float pdf = 1.0f;
        this->sampleLight(l_pos, pdf, sampler);
        auto obj2l = l_pos.coords - intersec.coords;
        auto l_o_dir = obj2l.normalized();
        Ray l_o(intersec.coords + EPSILON * intersec.normal, l_o_dir);
        auto r = obj2l.norm();
        auto r_pow = obj2l.x * obj2l.x + obj2l.y * obj2l.y + obj2l.z * obj2l.z;
        Intersection mid_intersec;
        this->intersect(l_o, mid_intersec);
        if (pdf > PDF_EPSILON && mid_intersec.happened && r <= mid_intersec.distance + 3 * EPSILON) {
            auto cos_theta = dotProduct(intersec.normal, l_o_dir);
            auto cos_theta_x = dotProduct(l_pos.normal, -l_o_dir);
            auto f_r = intersec.m->eval(r_o.direction, l_o_dir, intersec.normal);
            l_dir = f_r * l_pos.emit * cos_theta * cos_theta_x / (r_pow * pdf);
        }
        l += Vector3d::Max(coef * l_dir, Vector3d());

        if (sampler() > this->RussianRoulette)
            break;

        auto ray_o_dir = normalize(intersec.m->sample(ray.direction, intersec.normal, sampler));
        r_o = Ray(intersec.coords + EPSILON * ray_o_dir, ray_o_dir);
        auto pdf_indir = intersec.m->pdf(ray.direction, ray_o_dir, intersec.normal);
        if (pdf_indir < PDF_EPSILON) {
            break;
        }
        auto f_r = intersec.m->eval(ray.direction, ray_o_dir, intersec.normal);
        auto cos_theta = dotProduct(intersec.normal, ray_o_dir);
        coef *= f_r * cos_theta / pdf_indir / this->RussianRoulette;
    }
    return Vector3f(l.x, l.y, l.z);
}
