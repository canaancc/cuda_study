#include <stdint.h>
#include <stdio.h>

#include "keccak_256.cu"

__device__ int d_found_flag = 0;           // 全局标志：是否找到解
__device__ uint64_t d_found_nonce = 0;     // 找到的 nonce


__constant__ int ACCESSES = 8;
__constant__ int MIX_BYTES = 128;
__constant__ int HASH_BYTES = 64;
__constant__ int WORD_BYTES = 32;
__constant__ int DATASET_PARENTS = 16  ;

__device__ inline uint32_t fnv(uint32_t x, uint32_t y){
    return (x * 0x01000193) ^ y;
}

__device__ inline bool lessEq256(uint8_t* a, uint8_t* b){
    // 比较两个256位数据，每个数据由4个uint64_t组成
    // 从最高位开始比较（大端序）
    for(int i = 31; i >= 0; i--) {
        if(a[i] > b[i]) {
            return false;  // a > b
        }
        if(a[i] < b[i]) {
            return true;   // a < b
        }
        // 如果相等，继续比较下一个64位
    }
    return true;  // a == b，返回true（因为是lessEq，包含等于）
}

__device__ void keccak_256_kernal(const uint8_t* input, size_t in_len, uint8_t* output)

__device__ void eth_miner (uint64_t* dag, uint8_t* header, size_t header_len, int threadNum, int full_size, uint8_t* target ){

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
            }
            break;
        }
        nonce += 1;
        cnt += 1;
    }
}




__device__ void create_dag_item(uint32_t* cache, int index, int cache_size, int threadNum, uint32_t* dag){
     // 1. 获取初始cache项
    uint32_t mix[8]; 
    memcpy(mix, cache+index*8, 32);
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

    // 5. FNV混合多轮
    for(int round = 0; round < DATASET_PARENTS; round++) {
        uint32_t parent_index = fnv(index ^ round, mix[round % 8]) % (cache_size / 32);
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

