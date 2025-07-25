import numpy as np
import hashlib


R = np.array([
            [ 0, 36, 3, 41, 18],
            [ 1, 44,10, 45,  2],
            [62,  6,43, 15, 61],
            [28, 55,25, 21, 56],
            [27, 20,39,  8, 14]
           ], dtype=np.uint64)

RC = np.array([
  0x0000000000000001, 0x0000000000008082,
  0x800000000000808A, 0x8000000080008000,
  0x000000000000808B, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009,
  0x000000000000008A, 0x0000000000000088,
  0x0000000080008009, 0x000000008000000A,
  0x000000008000808B, 0x800000000000008B,
  0x8000000000008089, 0x8000000000008003,
  0x8000000000008002, 0x8000000000000080,
  0x000000000000800A, 0x800000008000000A,
  0x8000000080008081, 0x8000000000008080,
  0x0000000080000001, 0x8000000080008008
], dtype=np.uint64)


class Keccak:
    def __init__(self, rate=1088, capacity=512, output_len=256, d=0x06):
        self.rate = rate
        self.capacity = capacity
        self.block_size = rate // 8
        self.output_len = output_len
        self.d = d
        self.state = np.zeros((5,5), dtype=np.uint64)

    def rotl(self, x, n):
        bits = np.uint64(64)
        x = np.uint64(x)
        n = np.uint64(n % bits)
        bits = np.uint64(bits)
        mask = np.uint64(2**64 - 1)
        return np.bitwise_or(np.bitwise_and(np.left_shift(x, n), mask), np.right_shift(x, bits - n))

    def Keccak_round_function(self, A:np.ndarray, RC:int):
        A = A.astype(np.uint64)

        C = np.zeros(5, dtype=np.uint64)
        for i in range(5):
            C[i] = A[i,0] ^ A[i,1] ^ A[i,2] ^ A[i,3] ^ A[i,4]

        D = np.zeros(5, dtype=np.uint64)
        for i in range(5):
            D[i] = C[(i+4)%5] ^ self.rotl(C[(i+1)%5], 1)

        for i in range(5):
            for j in range(5):
                A[i,j] = A[i,j] ^ D[i]

        B = np.zeros((5,5), dtype=np.uint64)
        for i in range(5):
            for j in range(5):
                B[j,(2*i + 3*j)%5] = self.rotl(A[i,j], R[i,j])

        for i in range(5):
            for j in range(5):
                A[i,j] = B[i,j] ^ ((~B[(i+1)%5,j]) & B[(i+2)%5,j])

        A[0,0] = A[0,0] ^ RC
        return A


    def Keccak_f(self, b:int, A:np.ndarray):
        assert A.shape == (5,5), "Keccak shape must be (5,5)"
        assert b in [25,50,100,200,400,800,1600], "b must be 25,50,100,200,400,800,1600"
        w = b//25
        l = int(np.log2(w))
        n = 12 + 2* l
        for i in range(n):
            A = self.Keccak_round_function(A, RC[i])
        return A
    
    def pad10star1(self, msg: bytes, rate: int, d):
        """
        10*1 填充
        msg: 输入字节
        rate: 比特率 (bits)
        """
        block_size = rate // 8
        padded = bytearray(msg)
        padded.append(d)
        while len(padded) % block_size != block_size-1:
            padded.append(0x00)
        padded.append(0x80)

        return bytes(padded)

    def Absorb(self, padded: bytes, block_size: int, f_func):
        """
        吸收阶段
        padded: 填充后的字节
        block_size: 块大小 (bytes)
        """
        assert len(padded) % block_size == 0
        state = np.zeros((5,5), dtype=np.uint64)
        for i in range(len(padded)//block_size):
            block = padded[i*block_size:(i+1)*block_size]
            block_words = np.frombuffer(block, dtype=np.uint64) #按照 64 bit 进行分割
            for j in range(len(block_words)):
                x = j % 5
                y = j // 5
                state[x,y] ^= block_words[j]
            state = f_func(1600, state)

        return state

    def keccak_sponge(self, msg: bytes):
        """
        Keccak Sponge
        msg: 输入字节
        rate: 比特率 (bits)
        capacity: 容量 (bits)
        output_len: 输出长度 (bytes)
        f_func: permutation 函数 (比如 Keccak_f)
        """
        rate = self.rate
        capacity = self.capacity
        output_len = self.output_len
        d = self.d

        assert (rate + capacity) in [25,50,100,200,400,800,1600]
        block_size = rate // 8

        # 1. Padding
        padded = self.pad10star1(msg, rate, d)

        # 2. Absorb
        state = self.Absorb(padded, block_size, self.Keccak_f)

        # 3. Squeeze
        output_bytes = output_len//8
        Z = bytearray()
        while len(Z) < output_bytes:
            Z.extend(state.T.flatten()[:block_size].tobytes())
            if len(Z) >= output_bytes:
                break
            state = self.Keccak_f(1600, state)

        return bytes(Z[:output_bytes])


SHA3_256 = Keccak(rate=1088, capacity=512, output_len=256, d=0x06)

msg ="1234567890" 

if (hashlib.sha3_256(msg.encode()).hexdigest() == SHA3_256.keccak_sponge(msg.encode()).hex()):
    print("SHA3_256 Test Passed")
    print("SHA result: ", hashlib.sha3_256(msg.encode()).hexdigest())
else:
    print("SHA3_256 Test Failed")