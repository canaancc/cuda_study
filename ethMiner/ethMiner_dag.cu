#include <stdint.h>
#include <stdio.h>


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

