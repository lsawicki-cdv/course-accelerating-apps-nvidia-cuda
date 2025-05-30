#include <stdio.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  for (int i = 0; i < 5; ++i)
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    printNumber<<<2, 2, 0, stream>>>(i);
    cudaStreamDestroy(stream);
  }
  cudaDeviceSynchronize();
}
