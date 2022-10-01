//
// Created by goksu on 2/25/20.
// Modified by wangyingnan on 12.9.22
//

#include <fstream>
#include "render.cuh"
#include <curand_kernel.h>

__host__ __device__ inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);
    int m = 0;

    // change the spp value to change sample ammount
    int mask_L = (1<<msaa)-1;
    int mask_H = mask_L << msaa;
    int spp = std::max(this->spp, (1<<msaa)*(1<<msaa));
    double step = 1.0 / (1<<msaa);
    std::cout << "SPP: " << spp << "\n";
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) { 
            for (int k = 0; k < spp; k++){
                // generate primary ray direction
                int sub_j = (k & mask_H) >> msaa;
                int sub_i = k & mask_L;
                float x = (2 * (i + (0.5+sub_i)*step) / (float)scene.width - 1) *
                    imageAspectRatio * scale;
                float y = (1 - 2 * (j+ (0.5+sub_j)*step) / (float)scene.height) * scale;
                Vector3f dir = normalize(Vector3f(-x, y, 1));
                framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / double(spp);  
            }
            m++;
        }
        UpdateProgress(j / (float)scene.height);
    }
    UpdateProgress(1.f);

    // save framebuffer to file
    std::string filename = "cpu_"+ std::to_string(scene.width) + "x" + std::to_string(scene.height) + "_spp" + std::to_string(spp) + ".ppm";
    FILE* fp = fopen(filename.c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}

__global__ void render(SceneCuda* scene, int spp, int msaa, Vector3f* fb) {
    float scale = tan(deg2rad(scene->fov * 0.5));
    float imageAspectRatio = scene->width / (float)scene->height;
    Vector3f eye_pos(278, 273, -800);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= scene->width || j >= scene->height)
        return;

    int pixel_id = scene->width * j + i;
    curandState local_rand_state;
    curand_init(2022+pixel_id, 0, 0, &local_rand_state);
    auto sampler = [&local_rand_state]() {
        return curand_uniform(&local_rand_state);
    };

    int mask_L = (1<<msaa)-1;
    int mask_H = mask_L << msaa;
    spp = cuda::std::max(spp, (1<<msaa)*(1<<msaa));
    double step = 1.0 / (1<<msaa);
    Vector3f intensity;
    for (int k = 0; k < spp; k++){
        // generate primary ray direction
        int sub_j = (k & mask_H) >> msaa;
        int sub_i = k & mask_L;
        float x = (2 * (i + (0.5+sub_i)*step) / (float)scene->width - 1) * imageAspectRatio * scale;
        float y = (1 - 2 * (j+ (0.5+sub_j)*step) / (float)scene->height) * scale;

        // we should consider coordinate transformation between camera and world to get final ray direction
        Vector3f dir = normalize(Vector3f(-x, y, 1));
        intensity += scene->castRay(Ray(eye_pos, dir), sampler);  
    }
    fb[pixel_id] = intensity / double(spp);
}

__host__ void Renderer::RenderCuda(const Scene& scene) {
    int tx = 8;
    int ty = 8;
    {
        GpuMemAllocator gpu_alloctor;

        // allocate FB
        int num_pixels = scene.width*scene.height;
        size_t fb_size = num_pixels*sizeof(Vector3f);
        Vector3f *fb;
        fb = gpu_alloctor.allocManaged(&fb, fb_size);

        // copy scene to device
        auto s_cuda = scene.moveToDevice(gpu_alloctor);
        
        // deliver computation task to device
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
        dim3 blocks(cuda::std::ceil(double(scene.width)/tx), cuda::std::ceil(double(scene.height)/ty));
        dim3 threads(tx,ty);
        render<<<blocks, threads>>>(s_cuda, spp, msaa, fb);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // save framebuffer to file
        std::string filename = "cuda_"+ std::to_string(scene.width) + "x" + std::to_string(scene.height) + "_spp" + std::to_string(spp) + ".ppm";
        FILE* fp = fopen(filename.c_str(), "wb");
        (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
        for (auto i = 0; i < scene.height * scene.width; ++i) {
            static unsigned char color[3];
            color[0] = (unsigned char)(255 * cuda::std::pow(clamp(0, 1, fb[i].x), 0.6f));
            color[1] = (unsigned char)(255 * cuda::std::pow(clamp(0, 1, fb[i].y), 0.6f));
            color[2] = (unsigned char)(255 * cuda::std::pow(clamp(0, 1, fb[i].z), 0.6f));
            fwrite(color, 1, 3, fp);
        }
        fclose(fp);
    }
    cudaDeviceReset();
}