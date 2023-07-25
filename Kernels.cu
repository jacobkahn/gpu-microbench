
#include <device_launch_parameters.h>

#include "Kernels.cuh"

__global__ void stubKernel() {}

void stub() {
  stubKernel<<<1, 1>>>();
}
