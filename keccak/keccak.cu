#include <stdint.h>


__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
}; //ULL 表示无符号长整型

__constant__ uint64_t R[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14}
};

__device__ uint64_t rotl(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccakf(uint64_t state[5][5]) {
    uint64_t C[5];
    uint64_t D[5];
    uint64_t B[5][5];  // 修复：应该是二维数组
    
    for (int round = 0; round < 24; round++) {
        // θ (theta) step
        for (int x = 0; x < 5; x++) {
            C[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
        }

        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1);
        }

        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x][y] ^= D[x];
            }
        }

        // ρ (rho) and π (pi) steps
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                B[y][(2*x + 3*y) % 5] = rotl(state[x][y], R[x][y]);  // 修复：使用正确的数组索引
            }
        }

        // χ (chi) step
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }

        // ι (iota) step
        state[0][0] ^= RC[round];
    }
}

extern "C"
__global__ void keccak_kernel(uint64_t* input, uint64_t* output, int blocks) {  // 修复：拼写错误
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blocks) return;
    
    uint64_t state[5][5];
    
    // 从输入复制到state
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            state[i][j] = input[idx * 25 + i * 5 + j];
        }
    }
    
    // 执行Keccak-f[1600]
    keccakf(state);
    
    // 将结果复制到输出
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            output[idx * 25 + i * 5 + j] = state[i][j];
        }
    }
}



