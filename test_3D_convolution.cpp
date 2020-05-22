//Ref:
//https://stackoverflow.com/questions/22577857/3d-convolution-with-cuda-using-shared-memory
#include <iostream>

#define     MASK_WIDTH      3
#define     MASK_RADIUS     MASK_WIDTH / 2
#define     TILE_WIDTH      8
#define         W           (TILE_WIDTH + MASK_WIDTH - 1)

/**
 * GPU 2D Convolution using shared memory
 */
__global__ void convolution(float *I, float* M, float *P, int width, int height, int depth)
{
    /***** WRITE TO SHARED MEMORY *****/
    __shared__ float N_ds[W][W][W];

    // First batch loading
    int dest = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (threadIdx.z * TILE_WIDTH * TILE_WIDTH);
    int destTmp = dest;
    int destX = destTmp % W;
    destTmp = destTmp / W;
    int destY = destTmp % W;
    destTmp = destTmp / W;
    int destZ = destTmp;

    int srcZ = destZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    int srcY = destY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    int srcX = destX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    int src = srcX + (srcY * width) + (srcZ * width * height);

    if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destZ][destY][destX] = I[src];
    else
        N_ds[destZ][destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.x + (threadIdx.y * TILE_WIDTH) + (threadIdx.z * TILE_WIDTH * TILE_WIDTH) + TILE_WIDTH * TILE_WIDTH;
    destTmp = dest;
    destX = destTmp % W;
    destTmp = destTmp / W;
    destY = destTmp % W;
    destTmp = destTmp / W;
    destZ = destTmp;

    srcZ = destZ + (blockIdx.z * TILE_WIDTH) - MASK_RADIUS;
    srcY = destY + (blockIdx.y * TILE_WIDTH) - MASK_RADIUS;
    srcX = destX + (blockIdx.x * TILE_WIDTH) - MASK_RADIUS;
    src = srcX + (srcY * width) + (srcZ * width * height);

    if(destZ < W)
    {
        if(srcZ >= 0 && srcZ < depth && srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destZ][destY][destX] = I[src];
        else
            N_ds[destZ][destY][destX] = 0;
    }
    __syncthreads();

    /***** Perform Convolution *****/
    float sum = 0;
    int z;
    int y;
    int x;
    for(z = 0; z < MASK_WIDTH; z++)
        for(y = 0; y < MASK_WIDTH; y++)
            for(x = 0; x < MASK_WIDTH; x++)
                sum = sum + N_ds[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x] * M[x + (y * MASK_WIDTH) + (z * MASK_WIDTH * MASK_WIDTH)];
    z = threadIdx.z + (blockIdx.z * TILE_WIDTH);
    y = threadIdx.y + (blockIdx.y * TILE_WIDTH);
    x = threadIdx.x + (blockIdx.x * TILE_WIDTH);
    if(z < depth && y < height && x < width)
        P[x + (y * width) + (z * width * height)] = sum;

    __syncthreads();

}

int main(int argc, char* argv[])
{
    int image_width  = 16;
    int image_height = 16;
    int image_depth  = 5;

    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;

    float data[] =
    {
        1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f,  1.0f,  1.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f,  2.0f,  2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        3.0f,  3.0f,  3.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        4.0f,  4.0f,  4.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        5.0f,  5.0f,  5.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        6.0f,  6.0f,  6.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        7.0f,  7.0f,  7.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        8.0f,  8.0f,  8.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        9.0f,  9.0f,  9.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       10.0f, 10.0f, 10.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       11.0f, 11.0f, 11.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       12.0f, 12.0f, 12.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       13.0f, 13.0f, 13.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       14.0f, 14.0f, 14.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       15.0f, 15.0f, 15.0f, 1.0f, 3.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       16.0f, 16.0f, 16.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };

    float mask[] =
    {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };

    // CHECK CHECK CHECK CHECK CHECK
    int shared_memory_size = W * W * W;
    int block_size = TILE_WIDTH * TILE_WIDTH * TILE_WIDTH;
    int max_size = 3 * block_size;
    std::cout << "Block Size: " << block_size << " - Shared Memory Size: " << shared_memory_size << " - Max Size: " << max_size << std::endl;
    std::cout << "SHARED MEMORY SIZE HAS TO BE SMALLER THAN MAX SIZE IN ORDER TO WORK PROPERLY !!!!!!!";

    cudaMalloc((void **)&deviceInputImageData,  image_width * image_height * image_depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, image_width * image_height * image_depth * sizeof(float));
    cudaMalloc((void **)&deviceMaskData,        MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(float));

    cudaMemcpy(deviceInputImageData, data, image_width * image_height * image_depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,       mask, MASK_WIDTH  * MASK_WIDTH   * MASK_WIDTH  * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((image_width + TILE_WIDTH - 1) / TILE_WIDTH, (image_height + TILE_WIDTH - 1) / TILE_WIDTH, (image_depth + TILE_WIDTH - 1) / TILE_WIDTH);
    convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, image_width, image_height, image_depth);
    cudaDeviceSynchronize();

    cudaMemcpy(data, deviceOutputImageData, image_width * image_height * image_depth * sizeof(float), cudaMemcpyDeviceToHost);

    // Print data
    for(int i = 0; i < image_width * image_height * image_depth; ++i)
    {
        if((i % image_width) == 0)
            std::cout << std::endl;

        if((i % (image_width * image_height)) == 0)
            std::cout << std::endl;

        std::cout << data[i] << " - ";
    }

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    return 0;
}
