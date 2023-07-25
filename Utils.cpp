#include "Utils.h"

#include <sstream>
#include <stdexcept>

Timer Timer::start() {
  Timer t;
  t.startTime_ = std::chrono::high_resolution_clock::now();
  return t;
}

void sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

void sync(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

int getDevice() {
  int ret;
  CUDA_CHECK(cudaGetDevice(&ret));
  return ret;
}

void setDevice(int deviceId) {
  CUDA_CHECK(cudaSetDevice(deviceId));
}

namespace detail {

void check(cudaError_t err, const char* file, int line) {
  check(err, "", file, line);
}

void check(cudaError_t err, const char* prefix, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream ess;
    ess << prefix << '[' << file << ':' << line
        << "] CUDA error: " << cudaGetErrorString(err);
    throw std::runtime_error(ess.str());
  }
}

} // namespace detail

double
timeit(std::function<void()> fn, int executeIters = 100, int warmupIters = 10) {
  // warmup
  for (int i = 0; i < warmupIters; ++i) {
    fn();
  }
  sync();

  auto start = Timer::start();
  for (int i = 0; i < executeIters; i++) {
    fn();
  }
  sync();
  return Timer::stop(start) / executeIters;
}
