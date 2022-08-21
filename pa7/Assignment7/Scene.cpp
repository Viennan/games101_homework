//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"

const float PDF_EPSILON = 0.000005f;
const Vector3f ZERO3F = Vector3f();

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
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
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
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
        float tNearK = kInfinity;
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
    Intersection intersec = this->intersect(ray);
    if (!intersec.happened) {
        return Vector3f();
    }
    if (intersec.m->hasEmission()) {
        return depth == 0 ? dotProduct(intersec.normal, -ray.direction) * intersec.emit: Vector3f();
    }
    
    Vector3f l_dir;
    Intersection l_pos;
    float pdf = 1.0f;
    this->sampleLight(l_pos, pdf);
    auto obj2l = l_pos.coords - intersec.coords;
    auto l_o_dir = obj2l.normalized();
    Ray l_o(intersec.coords + EPSILON * intersec.normal, l_o_dir);
    auto r = obj2l.norm();
    auto r_pow = obj2l.x * obj2l.x + obj2l.y * obj2l.y + obj2l.z * obj2l.z;
    auto mid_intersec = this->intersect(l_o);
    if (pdf > PDF_EPSILON && mid_intersec.happened && r <= mid_intersec.distance + 3 * EPSILON) {
        auto cos_theta = dotProduct(intersec.normal, l_o_dir);
        auto cos_theta_x = dotProduct(l_pos.normal, -l_o_dir);
        auto f_r = intersec.m->eval(ray.direction, l_o_dir, intersec.normal);
        l_dir = f_r * l_pos.emit * cos_theta * cos_theta_x / (r_pow * pdf);
    }

    Vector3f l_indir;
    if (get_random_float() < this->RussianRoulette) {
        auto ray_o_dir = normalize(intersec.m->sample(ray.direction, intersec.normal));
        auto ray_o = Ray(intersec.coords + EPSILON * ray_o_dir, ray_o_dir);
        auto pdf_indir = intersec.m->pdf(ray.direction, ray_o_dir, intersec.normal);
        if (pdf_indir > PDF_EPSILON) {
            auto f_r = intersec.m->eval(ray.direction, ray_o_dir, intersec.normal);
            auto cos_theta = dotProduct(intersec.normal, ray_o_dir);
            l_indir = f_r * this->castRay(ray_o, depth + 1) * cos_theta / pdf_indir /this->RussianRoulette;
        }
    }

    return Vector3f::Max(l_dir, ZERO3F) + Vector3f::Max(l_indir, ZERO3F);    
}