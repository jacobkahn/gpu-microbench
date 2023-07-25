#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#define TIME(FUNC)                                           \
  std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

#define CUDA_CHECK(...) ::detail::check(__VA_ARGS__, __FILE__, __LINE__)

// From
// https://github.com/flashlight/flashlight/blob/main/flashlight/fl/common/Timer.h
class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;

 public:
  static Timer start();

  template <typename T = double>
  static T stop(const Timer& t) {
    return std::chrono::duration_cast<std::chrono::duration<T>>(
               std::chrono::high_resolution_clock::now() - t.startTime_)
        .count();
  }
};

void sync();

void sync(cudaStream_t stream);

int getDevice();

void setDevice(int deviceId);

namespace detail {

void check(cudaError_t err, const char* file, int line);

void check(cudaError_t err, const char* prefix, const char* file, int line);

} // namespace detail

/**
 * Benchmark an arbitrary closure
 */
double timeit(std::function<void()> fn);
