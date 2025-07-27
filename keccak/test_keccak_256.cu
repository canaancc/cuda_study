#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include "keccak_256.cu"

// 打印十六进制数据
void print_hex(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Keccak-256 CUDA 测试 ===\n\n");
    
    // 测试用例：字符串 "abc"
    const char* test_msg = "Hello";
    size_t len = strlen(test_msg);
    
    // 分配主机内存
    uint8_t* h_input = (uint8_t*)malloc(len);
    uint8_t* h_output = (uint8_t*)malloc(32);  // keccak-256输出32字节
    memcpy(h_input, test_msg, len);
    
    // 分配设备内存
    uint8_t* d_input;
    uint8_t* d_output;
    cudaMalloc(&d_input, len);
    cudaMalloc(&d_output, 32);
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, len, cudaMemcpyHostToDevice);
    
    // 启动kernel（1个block，1个thread）
    keccak_256_kernal<<<1, 1>>>(d_input, len, d_output);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ CUDA kernel错误: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 等待完成并复制结果
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, 32, cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("输入: \"%s\"\n", test_msg);
    printf("输出: ");
    print_hex(h_output, 32);
    printf("\n");
    
    // 清理内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("=== 测试完成 ===\n");
    
    return 0;
}