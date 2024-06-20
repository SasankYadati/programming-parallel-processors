__global__
void colorToGrayscaleKernel(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int gray_idx = row * width + col;
        int rgb_idx = gray_idx * 3;

        unsigned char r = Pin[rgb_idx];
        unsigned char g = Pin[rgb_idx + 1];
        unsigned char b = Pin[rgb_idx + 2];
        
        Pout[gray_idx] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}