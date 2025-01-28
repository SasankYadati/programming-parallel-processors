#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "vector_addition.h"
#include <cuda_runtime.h>

void benchmarkCPUImplementation(long long n, float *A_h, float *B_h, float *C_h) {
    double total_flops = n;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    vecAdd(A_h, B_h, C_h, n);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Vector addition on CPU: %f seconds\n", time);
    printf("CPU FLOPS (GHz): %f\n", total_flops / time / 1e9);
}

void benchmarkGPUImplementation(long long n, float *A_h, float *B_h, float *C_h) {
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, n * sizeof(float));
    cudaMalloc(&B_d, n * sizeof(float));
    cudaMalloc(&C_d, n * sizeof(float));
    cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int num_blocks = (int)ceilf(n / 1024.0f);


    cudaEvent_t start, end;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    vecAddKernel<<<num_blocks, threads_per_block>>>(A_d, B_d, C_d, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&milliseconds, start, end);
    double time = milliseconds / 1000.0;
    double total_flops = n;
    printf("Vector addition on GPU: %f seconds\n", time);
    printf("GPU FLOPS (GHz): %f\n", total_flops / time / 1e9);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    long long n = 5e8;
    float *A_h = (float *)malloc(n * sizeof(float));
    float *B_h = (float *)malloc(n * sizeof(float));
    float *C_h = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        A_h[i] = rand() % 100;
        B_h[i] = rand() % 100;
    }

    benchmarkCPUImplementation(n, A_h, B_h, C_h);

    benchmarkGPUImplementation(n, A_h, B_h, C_h);
    
    
    return 0;
}