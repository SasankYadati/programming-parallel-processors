#define BLUR_SIZE 1
#define WIDTH 12
#define HEIGHT 24
#include <stdio.h>

__global__
void blurKernel(unsigned char *in, unsigned char *out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < WIDTH && row < HEIGHT) {
        int avg_pixel_val = 0;
        int n_pixels = 0;
        for (int i=-BLUR_SIZE; i<=BLUR_SIZE; i++) {
            for (int j=-BLUR_SIZE; j<=BLUR_SIZE; j++) {
                int pixelRow = row + i;
                int pixelCol = col + j;
                if (pixelRow >= 0 && pixelRow < HEIGHT && pixelCol >= 0 && pixelCol < WIDTH) {
                    avg_pixel_val += in[pixelRow * WIDTH + pixelCol];
                    n_pixels++;
                }
            }
        }
        avg_pixel_val /= n_pixels;
        out[row * WIDTH + col] = (unsigned char)avg_pixel_val;
    }
}

void blur(unsigned char *in_h, unsigned char *out_h) {
    int size = WIDTH * HEIGHT * sizeof(unsigned char);
    unsigned char *in_d, *out_d;
    cudaMalloc((void **) &in_d, size);
    cudaMalloc((void **) &out_d, size);
    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(WIDTH/16.0), ceil(HEIGHT/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel<<<dimGrid, dimBlock>>>(in_d, out_d);
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

int main() {
    // create a char array that holds a dummy color image
    int size = WIDTH * HEIGHT;
    unsigned char *img = (unsigned char *)malloc(size * sizeof(unsigned char));
    printf("Original image:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx = i * WIDTH + j;
            img[idx] = rand() % 256;
            printf("%d ", img[idx]);
        }
        printf("\n");
    }
    unsigned char *img_blurred = (unsigned char *)malloc(size * sizeof(unsigned char));
    blur(img, img_blurred);
    printf("Blurred image:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx = i * WIDTH + j;
            printf("%d ", img_blurred[idx]);
        }
        printf("\n");
    }
}