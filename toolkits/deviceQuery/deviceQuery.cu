#include <cuda_runtime.h>
#include <iostream>
#include <string>

std::string getCudaArchString(int major, int minor) {
    // Combine major and minor into a single architecture version number
    int version = major * 10 + minor;

    switch (version) {
        case 30: return "sm_30";
        case 32: return "sm_32";
        case 35: return "sm_35";
        case 37: return "sm_37";
        case 50: return "sm_50";
        case 52: return "sm_52";
        case 53: return "sm_53";
        case 60: return "sm_60";
        case 61: return "sm_61";
        case 62: return "sm_62";
        case 70: return "sm_70";
        case 72: return "sm_72";
        case 75: return "sm_75";
        case 80: return "sm_80";
        case 86: return "sm_86";
        case 87: return "sm_87";
        case 89: return "sm_89";
        case 90: return "sm_90";
        default: return "unknown";
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::string cudaArch = getCudaArchString(deviceProp.major, deviceProp.minor);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  CUDA Capability: " << deviceProp.major << "." << deviceProp.minor << " (" << cudaArch << ")" << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Thread Dimensions: (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Concurrent Kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
