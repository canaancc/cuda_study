#include "keccak.cu"
#include <stdio.h>
#include <stdint.h>

// 256 keccak alg:
// rate = 1088
// capacity = 512

const int RATE = 136; // 1088 bit, 136 byte

__device__ void keccakf (uint64_t state[5][5]);


__device__ void absorb_block(uint64_t state[5][5], uint8_t* block, int len){
    uint64_t lane;
    for (int round = 0; round < (len/RATE) ; round ++){
        printf("len is %d \n", len);
        for (int index = 0; index < (RATE/8) ; index ++ ) {
            memcpy(&lane, block+ round*RATE+index*8, 8);

            int x = index % 5;
            int y = index / 5;

            state[x][y] ^= lane;
        }
        //for (int x = 0; x < 5; x ++){
        //    for(int y = 0; y < 5 ; y++){
        //        printf("state[%d][%d] is %llu \n", x, y, state[x][y]);
        //    }
        //}
        keccakf(state);
    }
}


__global__ void keccak_256_kernal(const uint8_t* input, size_t in_len, uint8_t* output){
    // initial state memory
    __shared__ uint64_t state[5][5]; // can be accces by gpu
    if(threadIdx.x == 0){
        //Only use threadIdx in case of competition
        for (int i = 0; i < 25 ; i++) {
            state[i%5][i/5] = 0;
        }
    }
    __syncthreads();

    // 1.Padding
    uint8_t padded[500] = {0}; // Max is 500 bytes
    memcpy(padded, input, in_len);
    //int extra_padd = in_len % RATE;
    int padded_len = ((in_len + RATE)/RATE) * RATE;

    padded[in_len] = 0x01;
    padded[padded_len -1] |= 0x80;

    //2. Absorb state
    absorb_block(state, padded, padded_len);

    //printf("== After absorb_block \n");
    for(int x = 0; x < 5; x ++){
        for(int y = 0; y < 5 ; y++){
            //printf("state[%d][%d] is 0x%016llx \n", x, y, state[x][y]);
        }
    }

    //3. Squeeze state
    uint8_t tmp[32];
    for (int i = 0; i < 4; i++){
        //printf("state[0][%d] is 0x%016llx \n", i, state[0][i]);
        memcpy(tmp+i*8, &state[i][0], 8);
    }
    

    memcpy(output, tmp, 32);
}
