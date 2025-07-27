#include <stdint.h>
#include <stdio.h>


__device__ int d_found_flag = 0;           // 全局标志：是否找到解
__device__ uint64_t d_found_nonce = 0;     // 找到的 nonce


// 主机端和设备端都可以使用的宏定义
#define ACCESSES 8
#define MIX_BYTES 128
#define HASH_BYTES 64
#define WORD_BYTES 32
#define DATASET_PARENTS 16
#define CACHE_ROUNDS 2
// 添加这行来包含 keccak 实现
#include "../keccak/keccak_256.cu"

// 添加函数声明
__global__ void eth_miner(uint32_t* dag, uint8_t* header, size_t header_len, int threadNum, int full_size, uint8_t* target);
__global__ void get_seed(int epoch_number, uint8_t* output_seed);
__global__ void create_cache(int cache_size, uint8_t* seed, uint32_t* cache);
__global__ void create_dag_item(uint32_t* cache, int cache_size, int full_size, uint32_t* dag);


// 辅助函数声明
__device__ inline uint32_t fnv(uint32_t x, uint32_t y);
__device__ inline bool lessEq256(uint8_t* a, uint8_t* b);


extern "C" {
    void eth_miner_wrapper(uint32_t* dag, uint8_t* header, size_t header_len, 
                          int threadNum, int full_size, uint8_t* target,
                          int* found_flag, uint64_t* found_nonce) {
        // 配置 CUDA 执行参数
        int blockSize = 256;
        int gridSize = (threadNum + blockSize - 1) / blockSize;
        
        // 调用 CUDA kernel
        eth_miner<<<gridSize, blockSize>>>(dag, header, header_len, threadNum, full_size, target);

        cudaDeviceSynchronize();

        // 将设备端结果复制到host端
        cudaMemcpyFromSymbol(found_flag, d_found_flag, sizeof(int));
        cudaMemcpyFromSymbol(found_nonce, d_found_nonce, sizeof(uint64_t));
    }
    
    void create_cache_wrapper(int cache_size, uint8_t* seed, uint32_t* cache) {
        // 调用 create_cache kernel
        create_cache<<<1, 1>>>(cache_size, seed, cache);
    }
    
    void create_dag_wrapper(uint32_t* cache, int cache_size, int full_size, uint32_t* dag) {
        int dataset_items = full_size / HASH_BYTES;
        int blockSize = 256;
        int gridSize = (dataset_items + blockSize - 1) / blockSize;
        
        // 修正：传递 full_size 而不是 dataset_items
        create_dag_item<<<gridSize, blockSize>>>(cache, cache_size, full_size, dag);
        
        // 添加同步和错误检查
        cudaDeviceSynchronize();
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("create_dag_item kernel failed: %s\n", cudaGetErrorString(error));
        }
    }
    void get_seed_wrapper(int epoch_number, uint8_t* d_seed){
        get_seed<<<1,1>>>(epoch_number, d_seed);
    }
}




__device__ inline uint32_t fnv(uint32_t x, uint32_t y){
    return (x * 0x01000193) ^ y;
}

__device__ inline bool lessEq256(uint8_t* a, uint8_t* b){
    // 比较两个256位数据（大端序）
    // 从最高位字节开始比较（索引0是最高位）
    for(int i = 0; i < 32; i++) {  // ✅ 正确：从高位到低位
        if(a[i] > b[i]) {
            return false;  // a > b
        }
        if(a[i] < b[i]) {
            return true;   // a < b
        }
        // 如果相等，继续比较下一个字节
    }
    return true;  // a == b，返回true（因为是lessEq，包含等于）
}


__global__ void eth_miner (uint32_t* dag, uint8_t* header, size_t header_len, int threadNum, int full_size, uint8_t* target ){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= threadNum){
        return;
    }

    uint64_t nonce = (UINT64_MAX/threadNum) * idx;
    uint64_t cnt = 0;
    uint8_t header_txt[200] = {0};
    memcpy(header_txt, header, header_len);


    /// START MINING ///

    while((atomicAdd(&d_found_flag, 0) == 0) && (cnt < (UINT64_MAX/threadNum))){ // check whether the result is found
        //combine nonce and header txt
        // 将 nonce 转换为字节数组（小端序）
        uint8_t nonce_bytes[8];
        for(int i = 0; i < 8; i++){
            nonce_bytes[i] = (nonce >> (i * 8)) & 0xFF;
        }
    
        // 反转 nonce 字节序（小端序 → 大端序）
        uint8_t nonce_reversed[8];
        for(int i = 0; i < 8; i++){
            nonce_reversed[i] = nonce_bytes[7 - i];
        }
    
        // 将反转后的 nonce 拼接到 header 后面
        for(int i = 0; i < 8; i++){
            header_txt[header_len + i] = nonce_reversed[i];
        }

        // 1. Get Mix
        uint8_t s_list[32]; // 256 bits
        keccak_256_kernal(header_txt, (header_len+8), s_list);

        uint32_t mix[MIX_BYTES/HASH_BYTES * 8] = {0}; 

        for (int i =0; i < MIX_BYTES/HASH_BYTES; i++){
            memcpy((uint8_t*)mix+i*32, s_list, 32);
        }

        uint32_t p = 0;
        //int w = MIX_BYTES / WORD_BYTES;
        int w = 2; // Todo: there is some chick point in w calculation
        int mix_hashes = MIX_BYTES / HASH_BYTES; // 2
        for( int i = 0; i < ACCESSES; i++){
            uint32_t tmp_xor = i^s_list[0];
            p = fnv(tmp_xor, mix[i%w]) % ((full_size / mix_hashes) * mix_hashes);

            // 1 dag data is 256 bits.
            uint32_t new_data[16] = {0};
            for (int j = 0; j < mix_hashes; j++){
                memcpy((uint8_t*)new_data+j*32, dag+p+j, 32); // dag data is 256 bits, 32 bytes
            }
            for (int j = 0 ; j < 16; j++){
                mix[j] = fnv(mix[j], new_data[j]);
            }
        }

        // compress Mix
        uint32_t cmix[4];
        for(int i = 0; i < 4; i++){
            cmix[i] = fnv(fnv(fnv(mix[i*4], mix[i*4+1]), mix[i*4+2]), mix[i*4+3]);
        }

        uint8_t final_seed[48];
        memcpy(final_seed, s_list, 32);
        for(int i = 0; i < 4; i++){
            final_seed[32 + i*4 + 0] = (cmix[i] >> 0) & 0xFF;
            final_seed[32 + i*4 + 1] = (cmix[i] >> 8) & 0xFF;
            final_seed[32 + i*4 + 2] = (cmix[i] >> 16) & 0xFF;
            final_seed[32 + i*4 + 3] = (cmix[i] >> 24) & 0xFF;
        }

        uint8_t final_result[32]; // final is 256 bits
        keccak_256_kernal(final_seed, 48, final_result);

        if(lessEq256(final_result, target)){
            if (atomicCAS(&d_found_flag, 0, 1) == 0) {  // 只有一个线程能成功
                d_found_nonce = nonce;  // 记录 nonce
                printf("找到 nonce: %llu\n", nonce);
                printf("final_result: ");
                for (int i = 0; i < 32; i++) {
                    printf("%02x", final_result[i]);
                }
                printf("\n");
                printf("thread id: %d\n", idx);
            }
            break;
        }
        nonce += 1;
        cnt += 1;
    }
}

__global__ void get_seed(int epoch_number, uint8_t* output_seed){

    uint8_t current_seed[32] = {0};  // 重命名避免冲突
    for (int i = 0; i < epoch_number; i++){
        keccak_256_kernal(current_seed, 32, current_seed);
    }
    memcpy(output_seed, current_seed, 32);
}


__global__ void create_cache(int cache_size, uint8_t* seed, uint32_t* cache){
    // 计算cache项的数量 (每项64字节)
    int n = cache_size / HASH_BYTES;  // HASH_BYTES = 64
    
    // 1. 顺序生成初始数据集
    // 第一项：直接对seed进行keccak_256
    uint8_t temp_hash[32];
    keccak_256_kernal(seed, 32, temp_hash);
    
    // 将第一个哈希结果转换为uint32_t数组并存储
    for(int i = 0; i < 8; i++) {
        cache[i] = (uint32_t)temp_hash[i*4] |
                  ((uint32_t)temp_hash[i*4+1] << 8) |
                  ((uint32_t)temp_hash[i*4+2] << 16) |
                  ((uint32_t)temp_hash[i*4+3] << 24);
    }
    
    // 后续项：对前一项进行keccak_256
    for(int i = 1; i < n; i++) {
        // 将前一项转换为字节数组
        uint8_t prev_bytes[32];
        for(int j = 0; j < 8; j++) {
            prev_bytes[j*4 + 0] = (cache[(i-1)*8 + j] >> 0) & 0xFF;
            prev_bytes[j*4 + 1] = (cache[(i-1)*8 + j] >> 8) & 0xFF;
            prev_bytes[j*4 + 2] = (cache[(i-1)*8 + j] >> 16) & 0xFF;
            prev_bytes[j*4 + 3] = (cache[(i-1)*8 + j] >> 24) & 0xFF;
        }
        
        // 计算keccak_256
        keccak_256_kernal(prev_bytes, 32, temp_hash);
        
        // 转换并存储
        for(int j = 0; j < 8; j++) {
            cache[i*8 + j] = (uint32_t)temp_hash[j*4] |
                            ((uint32_t)temp_hash[j*4+1] << 8) |
                            ((uint32_t)temp_hash[j*4+2] << 16) |
                            ((uint32_t)temp_hash[j*4+3] << 24);
        }
    }
    
    // 2. 使用低轮次的randmemohash
    for(int round = 0; round < CACHE_ROUNDS; round++) {
        for(int i = 0; i < n; i++) {
            // 获取当前项的第一个字节作为索引
            uint8_t first_byte = cache[i*8] & 0xFF;
            int v = first_byte % n;
            
            // 计算 o[(i-1+n) % n] XOR o[v]
            int prev_idx = ((i - 1 + n) % n) * 8;
            int v_idx = v * 8;
            
            uint32_t xor_result[8];
            for(int j = 0; j < 8; j++) {
                xor_result[j] = cache[prev_idx + j] ^ cache[v_idx + j];
            }
            
            // 将XOR结果转换为字节数组
            uint8_t xor_bytes[32];
            for(int j = 0; j < 8; j++) {
                xor_bytes[j*4 + 0] = (xor_result[j] >> 0) & 0xFF;
                xor_bytes[j*4 + 1] = (xor_result[j] >> 8) & 0xFF;
                xor_bytes[j*4 + 2] = (xor_result[j] >> 16) & 0xFF;
                xor_bytes[j*4 + 3] = (xor_result[j] >> 24) & 0xFF;
            }
            
            // 对XOR结果进行keccak_256
            keccak_256_kernal(xor_bytes, 32, temp_hash);
            
            // 将结果存回cache
            for(int j = 0; j < 8; j++) {
                cache[i*8 + j] = (uint32_t)temp_hash[j*4] |
                                ((uint32_t)temp_hash[j*4+1] << 8) |
                                ((uint32_t)temp_hash[j*4+2] << 16) |
                                ((uint32_t)temp_hash[j*4+3] << 24);
            }
        }
    }
}



__global__ void create_dag_item(uint32_t* cache, int cache_size, int full_size, uint32_t* dag){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dataset_items = full_size / HASH_BYTES;
    
    if (idx >= dataset_items) return;
    
    int index = idx;
    int cache_items = cache_size / HASH_BYTES;  // 计算 cache 项数
    
    // 1. 获取初始cache项 - 修正：使用模运算确保不越界
    uint32_t mix[8]; 
    int cache_index = index % cache_items;  // 确保索引在 cache 范围内
    memcpy(mix, cache + cache_index * 8, 32);
    
    // 2. 修改第一个元素
    mix[0] ^= index;

    // 3. 进行keccak_256哈希
    uint8_t mix_bytes[32];
    for(int i = 0; i < 8; i++) {
        mix_bytes[i*4 + 0] = (mix[i] >> 0) & 0xFF;
        mix_bytes[i*4 + 1] = (mix[i] >> 8) & 0xFF;
        mix_bytes[i*4 + 2] = (mix[i] >> 16) & 0xFF;
        mix_bytes[i*4 + 3] = (mix[i] >> 24) & 0xFF;
    }    
    // 4. 将哈希结果转回uint32_t数组
    uint8_t hash_result[32];
    keccak_256_kernal(mix_bytes, 32, hash_result);

    for(int i = 0; i < 8; i++) {
        mix[i] = (uint32_t)hash_result[i*4] |
                ((uint32_t)hash_result[i*4+1] << 8) |
                ((uint32_t)hash_result[i*4+2] << 16) |
                ((uint32_t)hash_result[i*4+3] << 24);
    }

    // 5. FNV混合多轮 - 修正：确保 parent_index 在 cache 范围内
    for(int round = 0; round < DATASET_PARENTS; round++) {
        uint32_t parent_index = fnv(index ^ round, mix[round % 8]) % cache_items;  // 使用 cache_items
        uint32_t parent[8];
        memcpy(parent, cache + parent_index * 8, 32);
        
        for(int i = 0; i < 8; i++) {
            mix[i] = fnv(mix[i], parent[i]);
        }
    }
     // 6. 最终keccak_256
    for(int i = 0; i < 8; i++) {
        mix_bytes[i*4 + 0] = (mix[i] >> 0) & 0xFF;
        mix_bytes[i*4 + 1] = (mix[i] >> 8) & 0xFF;
        mix_bytes[i*4 + 2] = (mix[i] >> 16) & 0xFF;
        mix_bytes[i*4 + 3] = (mix[i] >> 24) & 0xFF;
    }

    keccak_256_kernal(mix_bytes, 32, hash_result);

    // 7. 存储到DAG
    for(int i = 0; i < 8; i++) {
        dag[index * 8 + i] = (uint32_t)hash_result[i*4] |
                            ((uint32_t)hash_result[i*4+1] << 8) |
                            ((uint32_t)hash_result[i*4+2] << 16) |
                            ((uint32_t)hash_result[i*4+3] << 24);
    }
}

