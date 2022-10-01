#ifndef RAYTRACING_RENDER_H
#define RAYTRACING_RENDER_H

#include "scene.cuh"
struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

class Renderer
{
public:
    Renderer(): msaa(0) {};

    explicit Renderer(int spp, int msaa): spp(spp), msaa(msaa) {};


    void Render(const Scene& scene);

    __host__ void RenderCuda(const Scene& scene);

    int spp;
    int msaa;
private:
};

#endif