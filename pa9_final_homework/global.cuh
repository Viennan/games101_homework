#ifndef RAYTRACING_GLOBAL_H
#define RAYTRACING_GLOBAL_H

#include <iostream>
#include <random>
#include <cuda/std/cmath>
#include <unordered_map>
#include <tuple>

#undef M_PI
#define M_PI 3.141592653589793f

#define EPSILON 0.00016f

#define MAX_BVH_DEPTH 32

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

template<class T> 
__host__ __device__ inline T cuda_std_min(T x, T y) {
    return x < y ? x : y;
}

template<class T> 
__host__ __device__ inline T cuda_std_max(T x, T y) {
    return x > y ? x : y;
}

__host__ __device__ inline float clamp(const float &lo, const float &hi, const float &v)
{ return cuda_std_max(lo,cuda_std_min(hi, v)); }

__host__ __device__ inline bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = - 0.5 * b / a;
    else {
        float q = (b > 0) ?
                  -0.5 * (b + cuda::std::sqrt(discr)) :
                  -0.5 * (b - cuda::std::sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) cuda::std::swap(x0, x1);
    return true;
}

float get_random_float();

inline void UpdateProgress(float progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
};

class GpuMemAllocator {
public:
    GpuMemAllocator() = default;
    GpuMemAllocator(const GpuMemAllocator&) = delete;

    ~GpuMemAllocator() {
        release();
    };

    template<class T>
    T* allocManaged(T** p_ptr, size_t n) {
        checkCudaErrors(cudaMallocManaged(p_ptr, n));
        device_ptrs.push_back((void*)(*p_ptr));
        return *p_ptr;
    };

    template<class T>
    T* alloc(T** p_ptr, size_t n) {
        checkCudaErrors(cudaMalloc(p_ptr, n));
        device_ptrs.push_back((void*)(*p_ptr));
        return *p_ptr;
    };

    template<class T>
    void registerPtrPairs(T* host, T* device) {
        obj_map[(void*)host] = (void*)device;
    }

    template<class T>
    std::tuple<T*, bool> getDevicePtr(T* host) {
        auto iter = obj_map.find((void*)host);
        return iter == obj_map.end() ? std::tuple<T*, bool>(nullptr, false) :std::tuple<T*, bool>((T*)iter->second, true);
    }

    template<class T> 
    T* constructByCopy(T* host) {
        T* device;
        device = allocManaged(&device, sizeof(T));
        *device = *host;
        return device;
    }

    void release() {
        for (auto p: device_ptrs) {
            checkCudaErrors(cudaFree(p));
        }
        device_ptrs.clear();
        obj_map.clear();
    };

private:
    std::vector<void*> device_ptrs;
    std::unordered_map<void*, void*> obj_map;
};

#endif
