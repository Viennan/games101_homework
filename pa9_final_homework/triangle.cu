#include "triangle.cuh"
#include "OBJ_Loader.hpp"

__host__ MeshTriangle::MeshTriangle(const std::string& filename, Material *mt)
        : bvh(nullptr), bvh_cuda(nullptr), m(mt), triangles(nullptr), numTriangles(0), area(0.0f)
{
    objl::Loader loader;
    loader.LoadFile(filename);
    area = 0;
    assert(loader.LoadedMeshes.size() == 1);
    auto mesh = loader.LoadedMeshes[0];

    Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity()};
    Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};

    numTriangles = mesh.Vertices.size() / 3;
    triangles = (Triangle*)malloc(sizeof(Triangle) * numTriangles);
    for (size_t i = 0; i < numTriangles; ++i) {
        std::array<Vector3f, 3> face_vertices;
        size_t base = i * 3;
        for (int j = 0; j < 3; j++) {
            auto vert = Vector3f(mesh.Vertices[base + j].Position.X,
                                    mesh.Vertices[base + j].Position.Y,
                                    mesh.Vertices[base + j].Position.Z);
            face_vertices[j] = vert;

            min_vert = Vector3f(std::min(min_vert.x, vert.x),
                                std::min(min_vert.y, vert.y),
                                std::min(min_vert.z, vert.z));
            max_vert = Vector3f(std::max(max_vert.x, vert.x),
                                std::max(max_vert.y, vert.y),
                                std::max(max_vert.z, vert.z));
        }
        auto tri_i = new(triangles+i)Triangle(face_vertices[0], face_vertices[1],face_vertices[2], mt);
        area += tri_i->getArea();
    }
    bounding_box = Bounds3(min_vert, max_vert);
}

__host__ BVHAccel* MeshTriangle::buildBVH(int maxRecurDepth, uint32_t maxPrimsInNode) {
    if (maxRecurDepth <= 0) {
        throw std::runtime_error("MeshTriangle::buildBVH: max recursion depth should be greater than zero, but got " + std::to_string(maxRecurDepth));
    }
    std::vector<Object*> ptrs(this->numTriangles);
    std::iota(ptrs.begin(), ptrs.end(), this->triangles);
    if (this->bvh) {
        delete this->bvh;
        this->bvh = nullptr;
    }
    this->bvh = new BVHAccel(ptrs, maxPrimsInNode, maxRecurDepth);
    return bvh;
}

__global__ void new_cuda_triangle(Triangle* ptr, Vector3f v0, Vector3f v1, Vector3f v2, Material* m_device) {
    if (blockIdx.x==0&&threadIdx.x==0) {
        new(ptr)Triangle(v0, v1, v2, m_device);
    }
}

__host__ Triangle* Triangle::moveToDevice(GpuMemAllocator& alloctor) const {
    auto [m_device, flag] = alloctor.getDevicePtr(this->m);
    if (!flag) {
        m_device = alloctor.constructByCopy(this->m);
        alloctor.registerPtrPairs(m, m_device);
    }
    Triangle* d_ptr;
    d_ptr = alloctor.alloc(&d_ptr, sizeof(Triangle));
    new_cuda_triangle<<<1, 1>>>(d_ptr, v0, v1, v2, m_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return d_ptr;
}

__host__ __device__ MeshTriangle::~MeshTriangle() {
#ifndef __CUDA_ARCH__
    if (triangles) {
        free(triangles);
        triangles = nullptr;
    }
    if (bvh) {
        delete bvh;
        bvh = nullptr;
    }
#endif
}

struct TriangleData {
    Vector3f v0, v1, v2;
    Material* m;
};

__global__ void init_cuda_triangles(TriangleData* transData, Triangle* tris_device, size_t n) {
    auto step = gridDim.x * blockDim.x;
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    while(id < n) {
        const auto& t = transData[id];
        new (tris_device + id) Triangle(t.v0, t.v1, t.v2, t.m);
        id += step;
    }
}

__global__ void new_cuda_mesh_triangle(MeshTriangle* mesh_cuda, Bounds3 bbox, int numTriangles, 
    Triangle* tris_cuda, BVHAccelCuda* bvh_cuda, float area, Material* m_cuda)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        mesh_cuda = new(mesh_cuda)MeshTriangle();
        mesh_cuda->m = m_cuda;
        mesh_cuda->area = area;
        mesh_cuda->bounding_box = bbox;
        mesh_cuda->bvh = nullptr;
        mesh_cuda->bvh_cuda = bvh_cuda;
        mesh_cuda->triangles = tris_cuda;
        mesh_cuda->numTriangles = numTriangles;
    }
}

MeshTriangle* MeshTriangle::moveToDevice(GpuMemAllocator& alloctor) const {
    auto [m_device, flag] = alloctor.getDevicePtr(this->m);
    if (!flag) {
        m_device = alloctor.constructByCopy(this->m);
        alloctor.registerPtrPairs(m, m_device);
    }

    // move triangles to device by bvh order
    TriangleData* transData = nullptr;
    checkCudaErrors(cudaMallocManaged(&transData, numTriangles * sizeof(TriangleData)));
    const auto& bvh_prims = bvh->primitives;
    for (size_t i=0;i<numTriangles;++i) {
        auto tri = dynamic_cast<Triangle*>(bvh_prims[i]);
        transData[i].v0 = tri->v0;
        transData[i].v1 = tri->v1;
        transData[i].v2 = tri->v2;
        transData[i].m = m_device;
    }

    Triangle* tris_device;
    tris_device = alloctor.alloc(&tris_device, sizeof(Triangle)*numTriangles);
    uint32_t blocks = cuda_std_min<uint32_t>(cuda::std::ceil(double(numTriangles) / 64), 256);
    init_cuda_triangles<<<blocks, 64>>>(transData, tris_device, numTriangles);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(transData));

    // construct bvh in cuda
    Object** objects;
    objects = alloctor.allocManaged(&objects, sizeof(Object*)*numTriangles);
    for (size_t i=0;i<numTriangles;++i)
        objects[i] = tris_device + i;
    BVHAccelCuda* _bvh_cuda = bvh->moveToDevice(alloctor, objects);

    // assemble
    // When copying current MeshTriangle to gpu, we must init the new MeshTriangle instance in device code for correctly calling virtual function later on.
    MeshTriangle* mesh_cuda;
    mesh_cuda = alloctor.alloc(&mesh_cuda, sizeof(MeshTriangle));
    new_cuda_mesh_triangle<<<1, 1>>>(mesh_cuda, this->bounding_box, this->numTriangles, tris_device, _bvh_cuda, this->area, m_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return mesh_cuda;
}

void MeshTriangle::Locate(float x, float y, float z, float zoom) {
    Vector3f t(x, y, z);
    Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity()};
    Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    for (size_t i=0;i<numTriangles;++i) {
        const auto& tri = triangles[i];
        triangles[i] = Triangle(
            (tri.v0 - bounding_box.pMin) * zoom + t,
            (tri.v1 - bounding_box.pMin) * zoom + t,
            (tri.v2 - bounding_box.pMin) * zoom + t,
            tri.m
        );
    }
    area *= zoom;
    bounding_box = Bounds3(t, (bounding_box.pMax-bounding_box.pMin)*zoom + t);
}
