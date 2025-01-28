#ifndef VECTOR_ADDITION_H
#define VECTOR_ADDITION_H
#include <cuda_runtime.h>

void vecAdd(float *A_h, float *B_h, float *C_h, long long n);

__global__ void vecAddKernel(float *A_h, float *B_h, float *C_h, long long n);

#endif