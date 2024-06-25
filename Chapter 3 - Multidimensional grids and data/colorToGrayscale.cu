#include <stdio.h>
#define N_CHANNELS 3
#define WIDTH 12
#define HEIGHT 24

__global__
void colorToGrayscaleKernel(unsigned char *Pout, unsigned char *Pin) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < WIDTH && row < HEIGHT) {
        int gray_idx = row * WIDTH + col;
        int rgb_idx = gray_idx * N_CHANNELS;

        unsigned char r = Pin[rgb_idx];
        unsigned char g = Pin[rgb_idx + 1];
        unsigned char b = Pin[rgb_idx + 2];

        Pout[gray_idx] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

int nvalsIn() {
    return WIDTH * HEIGHT * N_CHANNELS;
}

int nvalsOut() {
    return WIDTH * HEIGHT;
}

void colorToGrayscale(unsigned char *Pout_h, unsigned char *Pin_h) {
    int n_vals_in = nvalsIn();
    int n_vals_out = nvalsOut();
    unsigned char *Pout_d, *Pin_d;
    cudaMalloc((void **) &Pout_d, n_vals_out * sizeof(unsigned char));
    cudaMalloc((void **) &Pin_d, n_vals_in * sizeof(unsigned char));
    cudaMemcpy(Pin_d, Pin_h, n_vals_in * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(WIDTH/16.0), ceil(HEIGHT/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d);

    cudaMemcpy(Pout_h, Pout_d, n_vals_out * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(Pout_d);
    cudaFree(Pin_d);
}

void populateImageArray(unsigned char *img) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int gray_idx = i * WIDTH + j;
            int rgb_idx = gray_idx * N_CHANNELS;
            // fill with random pixel vals
            img[rgb_idx] = rand() % 256;
            img[rgb_idx + 1] = rand() % 256;
            img[rgb_idx + 2] = rand() % 256;
        }
    }
}

int main() {
    // create a char array that holds a dummy color image
    int n_vals_in = nvalsIn();
    int n_vals_out = nvalsOut();
    unsigned char *img = (unsigned char *)malloc(n_vals_in * sizeof(unsigned char));
    populateImageArray(img);
    printf("Original image:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int gray_idx = i * WIDTH + j;
            int rgb_idx = gray_idx * N_CHANNELS;
            printf("(%d, %d, %d)\t", img[rgb_idx], img[rgb_idx + 1], img[rgb_idx + 2]);
        }
        printf("\n");
    }
    unsigned char *img_grayscaled = (unsigned char *)malloc(n_vals_out * sizeof(unsigned char));
    colorToGrayscale(img_grayscaled, img);

    printf("Grayscaled image check:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int gray_idx = i * WIDTH + j;
            int rgb_idx = gray_idx * N_CHANNELS;

            unsigned char r = img[rgb_idx];
            unsigned char g = img[rgb_idx + 1];
            unsigned char b = img[rgb_idx + 2];

            unsigned char grayscaled = 0.21f * r + 0.72f * g + 0.07f * b;
            printf("%d ", img_grayscaled[gray_idx] - grayscaled);
        }
        printf("\n");
    }
}