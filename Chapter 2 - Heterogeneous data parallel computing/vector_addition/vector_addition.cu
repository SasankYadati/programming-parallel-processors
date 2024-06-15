#include <stdio.h>

__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // allocate device memory
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // copy host memory to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
        
    // launch kernel for vector addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // copy device memory to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int n = 1024;
    
    // create arrays of floats of size n
    float *A = (float *)malloc(n * sizeof(float));
    float *B = (float *)malloc(n * sizeof(float));
    float *C = (float *)malloc(n * sizeof(float));
    
    // initialize arrays
    for (int i = 0; i < n; i++) {
        A[i] = i;
    }
    for (int i = 0; i < n; i++) {
        B[i] = i;
    }

    vecAdd(A, B, C, n);
    
    // print array C
    for (int i = 0; i < n; i++) {
        printf("%f ", C[i]);
    }
}