#include "render.cuh"
#include "scene.cuh"
#include "triangle.cuh"
#include "vector.cuh"
#include "global.cuh"
#include <chrono>

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char** argv)
{
    constexpr int max_bvh_depth = 20;
    constexpr int max_prims_in_node = 4;
    constexpr int msaa = 3;
    constexpr int spp = 10240;

    // Change the definition here to change resolution
    Scene scene(1024, 1024);

    Material* red = new Material(DIFFUSE, Vector3f(0.0f));
    red->Kd = Vector3f(0.63f, 0.065f, 0.05f);
    Material* green = new Material(DIFFUSE, Vector3f(0.0f));
    green->Kd = Vector3f(0.14f, 0.45f, 0.091f);
    Material* white = new Material(DIFFUSE, Vector3f(0.0f));
    white->Kd = Vector3f(0.725f, 0.71f, 0.68f);
    Material* yellow = new Material(DIFFUSE, Vector3f(0.0f));
    yellow->Kd = Vector3f(0.8f, 0.59f, 0.05f);
    Material* blue = new Material(DIFFUSE, Vector3f(0.0f));
    blue->Kd = Vector3f(0.055f, 0.329f, 0.745f);

    Material* light = new Material(DIFFUSE, (8.0f * Vector3f(0.747f+0.058f, 0.747f+0.258f, 0.747f) + 15.6f * Vector3f(0.740f+0.287f,0.740f+0.160f,0.740f) + 18.4f *Vector3f(0.737f+0.642f,0.737f+0.159f,0.737f)));
    light->Kd = Vector3f(0.65f);
    Material* reflect = new Material(REFLECT, Vector3f(0.0f));
    reflect->Ks = Vector3f(0.95f);

    // Material* hightlight = new Material(HIGHLIGHT, Vector3f(0.0f));
    // hightlight->Ks = Vector3f(0.95f);
    // hightlight->maxSpecularAngle = 10.0/180.0*M_PI;
    // hightlight->specularExponent = 10;

    MeshTriangle floor("../models/cornellbox/floor.obj", white);
    MeshTriangle shortbox("../models/cornellbox/shortbox.obj", yellow);
    MeshTriangle tallbox("../models/cornellbox/tallbox.obj", reflect);
    MeshTriangle left("../models/cornellbox/left.obj", red);
    MeshTriangle right("../models/cornellbox/right.obj", green);
    MeshTriangle light_("../models/cornellbox/light.obj", light);
    MeshTriangle bunny("../models/bunny/bunny.obj", blue);
    bunny.Locate(150, 165.001, 85, 1000);

    scene.Add(&floor);
    scene.Add(&shortbox);
    scene.Add(&tallbox);
    scene.Add(&left);
    scene.Add(&light_);
    scene.Add(&right);
    scene.Add(&bunny);

    scene.buildBVH(max_bvh_depth, max_prims_in_node);

    Renderer r(spp, msaa);

    auto start = std::chrono::system_clock::now();
    // r.Render(scene);
    r.RenderCuda(scene);
    auto stop = std::chrono::system_clock::now();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}