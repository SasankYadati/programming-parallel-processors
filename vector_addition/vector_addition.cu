#include "vector_addition.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vecAddKernel(float *A_h, float *B_h, float *C_h, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C_h[idx] = A_h[idx] + B_h[idx];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, long long n) {
    for (long long i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}