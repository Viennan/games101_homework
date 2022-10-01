//
// Created by LEI XU on 5/16/19.
// Modified by Wang Yingnan on 8/23/22
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "vector.cuh"

enum MaterialType { DIFFUSE,REFLECT,HIGHLIGHT};

// 48Bytes, naturally aligned
__align__(8) class Material{
private:

    // Compute reflection direction
    __host__ __device__ Vector3f reflect(const Vector3f &I, const Vector3f &N) const
    {
        return I - 2 * dotProduct(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and negate the normal N
    __host__ __device__ Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        Vector3f n = N;
        if (cosi < 0) { cosi = -cosi; } else { cuda::std::swap(etai, etat); n= -N; }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - cuda::std::sqrtf(k)) * n;
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    __host__ __device__ void fresnel(const Vector3f &I, const Vector3f &N, const float &ior, float &kr) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0) {  cuda::std::swap(etai, etat); }
        // Compute sini using Snell's law
        float sint = etai / etat * cuda::std::sqrtf(cuda_std_max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1) {
            kr = 1;
        }
        else {
            float cost = cuda::std::sqrtf(cuda_std_max(0.f, 1 - sint * sint));
            cosi = cuda::std::fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is given by:
        // kt = 1 - kr;
    }

    __host__ __device__ Vector3f toWorld(const Vector3f &a, const Vector3f &N){
        Vector3f B, C;
        if (cuda::std::fabs(N.x) > cuda::std::fabs(N.y)){
            float invLen = 1.0f / cuda::std::sqrt(N.x * N.x + N.z * N.z);
            C = Vector3f(N.z * invLen, 0.0f, -N.x *invLen);
        }
        else {
            float invLen = 1.0f / cuda::std::sqrt(N.y * N.y + N.z * N.z);
            C = Vector3f(0.0f, N.z * invLen, -N.y *invLen);
        }
        B = crossProduct(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

public:
    MaterialType m_type;
    //Vector3f m_color;
    Vector3f m_emission;
    float ior;
    Vector3f Kd, Ks;
    float specularExponent;
    float maxSpecularAngle;
    //Texture tex;

    __host__ __device__ inline Material(MaterialType t=DIFFUSE, Vector3f e=Vector3f(0,0,0));
    __host__ __device__ Material(const Material& m) = default;
    __host__ __device__ inline MaterialType getType();
    //inline Vector3f getColor();
    __host__ __device__ inline Vector3f getColorAt(double u, double v);
    __host__ __device__ inline Vector3f getEmission();
    __host__ __device__ inline bool hasEmission();

    // sample a ray by Material properties
    template<class Sampler>
    __host__ __device__ inline Vector3f sample(const Vector3f &wi, const Vector3f &N, Sampler sampler);
    // given a ray, calculate the PdF of this ray
    __host__ __device__ inline float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);
    // given a ray, calculate the contribution of this ray
    __host__ __device__ inline Vector3f eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);

};

__host__ __device__ Material::Material(MaterialType t, Vector3f e){
    m_type = t;
    //m_color = c;
    m_emission = e;
}

__host__ __device__ MaterialType Material::getType(){return m_type;}
///Vector3f Material::getColor(){return m_color;}
__host__ __device__ Vector3f Material::getEmission() {return m_emission;}
__host__ __device__ bool Material::hasEmission() {
    if (m_emission.norm() > EPSILON) return true;
    else return false;
}

__host__ __device__ Vector3f Material::getColorAt(double u, double v) {
    return Vector3f();
}

// sampler should be a callable object which returns a float number between [0, 1]
template<class Sampler>
__host__ __device__ Vector3f Material::sample(const Vector3f &wi, const Vector3f &N, Sampler sampler){
    switch(m_type){
        case DIFFUSE:
        {
            // uniform sample on the hemisphere
            float x_1 = sampler(), x_2 = sampler();
            float z = cuda::std::fabs(1.0f - 2.0f * x_1);
            float r = cuda::std::sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
            Vector3f localRay(r*cuda::std::cos(phi), r*cuda::std::sin(phi), z);
            return toWorld(localRay, N);
            
            break;
        }
        case REFLECT:
        {
            return reflect(wi, N).normalized();
        }
        case HIGHLIGHT:
        {
            auto v_reflect = reflect(wi, N).normalized();
            float x = sampler(), y = sampler();
            auto v_random = Vector3f(x, y, cuda::std::sqrt(1.0f-x*x-y*y));
            auto v_axis = crossProduct(v_reflect, v_random);
            auto v_t = crossProduct(v_axis, v_reflect);
            auto rot = sampler() * maxSpecularAngle;
            return (v_reflect * cuda::std::cosf(rot) + v_t * cuda::std::sinf(rot)).normalized();
        }
    }
}

__host__ __device__ float Material::pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
    switch(m_type){
        case DIFFUSE:
        {
            // uniform sample probability 1 / (2 * PI)
            if (dotProduct(wo, N) > 0.0f)
                return 0.5f / M_PI;
            else
                return 0.0f;
            break;
        }
        case REFLECT:
        {
            return ((reflect(wi, N) - wo).norm() > 2 * EPSILON) ? 0.0f : 1.0f;
        }
        case HIGHLIGHT:
        {
            return 1.0f / 2*M_PI*(1-cuda::std::cosf(maxSpecularAngle));
        }
    }
}

__host__ __device__ Vector3f Material::eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
    switch(m_type){
        case DIFFUSE:
        {
            // calculate the contribution of diffuse   model
            float cosalpha = dotProduct(N, wo);
            if (cosalpha > 0.0f) {
                Vector3f diffuse = Kd / M_PI;
                return diffuse;
            }
            else
                return Vector3f(0.0f);
            break;
        }
        case REFLECT:
        {
            return ((reflect(wi, N) - wo).norm() > 2 * EPSILON) ? Vector3f() : Ks;
        }
        case HIGHLIGHT:
        {
            float cosaplpha = dotProduct(N, wo);
            if (cosaplpha > 0.0f) {
                return Kd/M_PI + Ks * cuda::std::powf(cosaplpha, specularExponent);
            }
            else {
                return Vector3f(0.0f);
            }
        }
    }
}

#endif //RAYTRACING_MATERIAL_H