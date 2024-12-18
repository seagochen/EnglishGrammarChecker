#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)
#endif

// 简单的计时函数，用cudaEvent
float measureCopyTime(void* dst, const void* src, size_t size, cudaMemcpyKind kind, int repeats=10) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热一次
    CHECK_CUDA(cudaMemcpy(dst, src, size, kind));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < repeats; i++) {
        CHECK_CUDA(cudaMemcpy(dst, src, size, kind));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // 返回单次拷贝时间，单位：毫秒
    return ms / repeats;
}

int main(int argc, char** argv) {
    // 测试数据大小(字节)
    // 可以自行调整，比如 100 MB = 100 * 1024 * 1024 bytes
    size_t dataSize = 100 * 1024 * 1024; 

    // 分配主机内存
    void* h_src = malloc(dataSize);
    void* h_dst = malloc(dataSize);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // 初始化主机数据
    memset(h_src, 1, dataSize);
    memset(h_dst, 0, dataSize);

    // 分配设备内存
    void* d_src;
    void* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, dataSize));
    CHECK_CUDA(cudaMalloc(&d_dst, dataSize));

    // ===========================
    // Host to Device 带宽测试
    // ===========================
    float h2d_time = measureCopyTime(d_dst, h_src, dataSize, cudaMemcpyHostToDevice);
    float h2d_bandwidth = (dataSize * 1e-9f) / (h2d_time * 1e-3f); // GB/s
    printf("Host to Device: %.3f GB/s\n", h2d_bandwidth);

    // ===========================
    // Device to Host 带宽测试
    // ===========================
    float d2h_time = measureCopyTime(h_dst, d_dst, dataSize, cudaMemcpyDeviceToHost);
    float d2h_bandwidth = (dataSize * 1e-9f) / (d2h_time * 1e-3f); // GB/s
    printf("Device to Host: %.3f GB/s\n", d2h_bandwidth);

    // ===========================
    // Device to Device 带宽测试
    // ===========================
    float d2d_time = measureCopyTime(d_dst, d_src, dataSize, cudaMemcpyDeviceToDevice);
    float d2d_bandwidth = (dataSize * 1e-9f) / (d2d_time * 1e-3f); // GB/s
    printf("Device to Device: %.3f GB/s\n", d2d_bandwidth);

    // 清理内存
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return 0;
}
