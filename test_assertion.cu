//ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion

#include <assert.h>

__global__ void testAssert(void)
{
    int is_one = 1;
    int should_be_one = 0;

    // This will have no effect
    assert(is_one);

    // This will halt kernel execution
    assert(should_be_one);
}

int main(int argc, char* argv[])
{
    testAssert<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}

//[Note]:
//Assertions are for debugging purposes. 
//They can affect performance and it is therefore recommended to disable them in production code. 
//They can be disabled at compile time by defining the NDEBUG preprocessor macro before including assert.h. 
//Note that expression should not be an expression with side effects (something like (++i > 0), for example), 
//otherwise disabling the assertion will affect the functionality of the code.
