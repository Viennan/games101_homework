cmake_minimum_required(VERSION 3.24)

project(CudaRayTracing VERSION 1.0 LANGUAGES CXX CUDA)

set (CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_BUILD_TYPE "Release")

set(SOURCE_FILE "main.cu;render.cu;scene.cu;bvh.cu;triangle.cu;global.cu")
file(GLOB SOUREC_HEAD_FILE CONFIGURE_DEPENDS "*.hpp" "*.h" "*.cuh")

add_executable(${PROJECT_NAME} ${SOURCE_FILE} ${SOURCE_HEAD_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "61")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Debug>:-G>")