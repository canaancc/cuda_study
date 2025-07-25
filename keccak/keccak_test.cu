#include <stdio.h>
#include <cuda_runtime.h>
#include "keccak.cu"


// 测试代码
int main() {
    const int num_blocks = 2;  // 测试2个数据块
    const int state_size = 25; // 5x5 = 25个uint64_t
    
    // 主机内存分配
    uint64_t* h_input = new uint64_t[num_blocks * state_size];
    uint64_t* h_output = new uint64_t[num_blocks * state_size];
    
    // 初始化测试数据
    for (int block = 0; block < num_blocks; block++) {
        for (int i = 0; i < state_size; i++) {
            h_input[block * state_size + i] = block * 100 + i;  // 简单的测试数据
        }
    }
    printf("=== Keccak测试输入 ===\n");
    for (int block = 0; block < num_blocks; block++) {
        printf("\nBlock %d:\n", block);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%016llx ", h_input[block * 25 + i * 5 + j]);
            }
        }
    }
    
    // 设备内存分配
    uint64_t* d_input;
    uint64_t* d_output;
    
    cudaMalloc(&d_input, num_blocks * state_size * sizeof(uint64_t));
    cudaMalloc(&d_output, num_blocks * state_size * sizeof(uint64_t));
    
    // 数据传输到设备
    cudaMemcpy(d_input, h_input, num_blocks * state_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 配置CUDA执行参数
    int threads_per_block = 256;
    int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;
    
    printf("启动CUDA kernel: %d blocks, %d threads per block\n", blocks_per_grid, threads_per_block);
    
    // 启动kernel
    keccak_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, num_blocks);
    
    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel执行错误: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 等待kernel完成
    cudaDeviceSynchronize();
    
    // 将结果传输回主机
    cudaMemcpy(h_output, d_output, num_blocks * state_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("\n=== Keccak测试结果 ===\n");
    for (int block = 0; block < num_blocks; block++) {
        printf("\nBlock %d:\n", block);
        printf("输入:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%016llx ", h_input[block * 25 + i * 5 + j]);
            }
            printf("\n");
        }
        
        printf("输出:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%016llx ", h_output[block * 25 + i * 5 + j]);
            }
            printf("\n");
        }
    }
    
    // 清理内存
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n测试完成！\n");
    return 0;
}
