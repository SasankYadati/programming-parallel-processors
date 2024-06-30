#include <stdio.h>
#define M 20
#define N 20
#define P 20

__global__
void matrixMultiplicationKernel(float *A, float *B, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < P) {
        float sum = 0.0f;
        for (int k=0; k<N; k++) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}


void matrixMultiplication(float *A_h, float *B_h, float *C_h) {
    float *A_d, *B_d, *C_d;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);
    cudaMalloc((void **) &A_d, sizeA);
    cudaMalloc((void **) &B_d, sizeB);
    cudaMalloc((void **) &C_d, sizeC);

    cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);

    // one thread per output element
    // blocks of 16 x 16 threads
    // grid of ceil(M/16.0) x ceil(P/16.0)
    dim3 dimGrid(ceil(M/16.0), ceil(P/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);

    cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    float A[M * N];
    float B[N * P];
    float C[M * P];

    // identity matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 0.0f;
        }
        A[i * N + i] = 1.0f;
    }

    // first N x P positive integers
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            B[i * P + j] = i * P + j;
        }
    }

    matrixMultiplication(A, B, C);

    // should be same as B_h
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            printf("%f ", C[i * P + j]);
        }
        printf("\n");
    }
    return 0;
}