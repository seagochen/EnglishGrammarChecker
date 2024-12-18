#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 一个简单的核函数，用来对数据进行读写操作
__global__ void kernelReadWrite(int *data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int val = data[idx];  // 读
        data[idx] = val + 1;  // 写
    }
}

// 测量内核执行时间的辅助函数
float runKernelAndMeasure(int *data, size_t N, bool sync = true) {
    // 创建event用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);

    CHECK_CUDA(cudaEventRecord(start, 0));
    kernelReadWrite<<<blocks, threads>>>(data, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    if (sync) {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    return ms;
}

int main(int argc, char **argv) {
    // 数据大小 (以整型元素计)
    size_t N = 100 * 1024 * 1024; // 100M ints左右
    if (argc > 1) {
        N = atol(argv[1]);
    }
    size_t dataSize = N * sizeof(int);

    // 检查设备数量并选择设备
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        CHECK_CUDA(cudaSetDevice(0));
    } else {
        fprintf(stderr, "No CUDA device found.\n");
        return -1;
    }

    // 打印设备信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using device 0: %s, totalGlobalMem = %zu bytes\n", prop.name, prop.totalGlobalMem);

    // 使用统一内存分配
    int *data;
    CHECK_CUDA(cudaMallocManaged(&data, dataSize));

    // 主机初始化数据
    for (size_t i = 0; i < N; i++) {
        data[i] = i % 100;
    }

    // 将设备设为首选访问位置（可选，不一定必要）
    CHECK_CUDA(cudaMemAdvise(data, dataSize, cudaMemAdviseSetPreferredLocation, 0));

    // 第一次内核访问：数据在主机侧，需要迁移到GPU
    float time_first = runKernelAndMeasure(data, N);
    printf("First kernel run (no prefetch): %.3f ms\n", time_first);

    // 第二次内核访问：数据已经在GPU
    float time_second = runKernelAndMeasure(data, N);
    printf("Second kernel run (data resident on GPU): %.3f ms\n", time_second);

    // 主机再次访问数据，模拟CPU侧对数据的访问
    for (size_t i = 0; i < N; i += (N/100)) {
        volatile int tmp = data[i]; // 强制访问
        (void)tmp;
    }

    // 再次运行内核访问，可能又有迁移开销
    float time_third = runKernelAndMeasure(data, N);
    printf("Third kernel run (after host touch): %.3f ms\n", time_third);

    // 预取数据到GPU，降低下一次访问的延迟
    CHECK_CUDA(cudaMemPrefetchAsync(data, dataSize, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 再次访问数据，这次应该更快了
    float time_prefetched = runKernelAndMeasure(data, N);
    printf("Kernel run after prefetch: %.3f ms\n", time_prefetched);

    // 清理内存
    CHECK_CUDA(cudaFree(data));

    return 0;
}
