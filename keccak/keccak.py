import hashlib
import numpy as np

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


def Keccak_f(b:int, A:np.ndarray):
    assert A.shape == (5,5), "Keccak shape must be (5,5)"
    assert b in [25,50,100,200,400,800,1600], "b must be 25,50,100,200,400,800,1600"
    w = b//25
    l = int(np.log2(w))
    n = 12 + 2* l
    for i in range(n):
        A = Keccak_round_function(A, RC[i])
    return A


def rotl(x, n):
    import numpy as np
    bits = np.uint64(64)
    x = np.uint64(x)
    n = np.uint64(n % bits)
    bits = np.uint64(bits)
    mask = np.uint64(2**64 - 1)
    return np.bitwise_or(np.bitwise_and(np.left_shift(x, n), mask), np.right_shift(x, bits - n))

def Keccak_round_function(A:np.ndarray, RC:int):
    A = A.astype(np.uint64)

    C = np.zeros(5, dtype=np.uint64)
    for i in range(5):
        C[i] = A[i,0] ^ A[i,1] ^ A[i,2] ^ A[i,3] ^ A[i,4]

    D = np.zeros(5, dtype=np.uint64)
    for i in range(5):
        D[i] = C[(i+4)%5] ^ rotl(C[(i+1)%5], 1)

    for i in range(5):
        for j in range(5):
            A[i,j] = A[i,j] ^ D[i]

    B = np.zeros((5,5), dtype=np.uint64)
    for i in range(5):
        for j in range(5):
            B[j,(2*i + 3*j)%5] = rotl(A[i,j], R[i,j])

    for i in range(5):
        for j in range(5):
            A[i,j] = B[i,j] ^ ((~B[(i+1)%5,j]) & B[(i+2)%5,j])

    A[0,0] = A[0,0] ^ RC

    return A


def pad10star1(msg: bytes, rate: int, d=0x01):
    """
    Keccak 10*1 填充（修复版）
    msg: 输入字节
    rate: 比特率 (bits)
    d: 域分隔符，对于keccak-256是0x01
    """
    block_size = rate // 8  # 对于keccak-256，rate=1088，block_size=136
    padded = bytearray(msg)
    
    # 1. 添加域分隔符
    padded.append(d)
    
    # 2. 计算需要填充到的总长度（向上取整到block_size的倍数）
    padded_len = ((len(padded) + block_size - 1) // block_size) * block_size
    
    # 3. 填充零字节
    while len(padded) < padded_len:
        padded.append(0x00)
    
    # 4. 设置最后一个字节的最高位（处理边界情况）
    if len(padded) > 0:
        padded[-1] |= 0x80
    
    return bytes(padded)

def Absorb(padded: bytes, block_size: int, f_func):
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
       # for x in range(5):
           # for y in range(5):
               # print(f"state[{x}][{y}] is {state[x][y]}")
        state = f_func(1600, state)

    return state
        


def keccak_sponge(msg: bytes, rate: int, capacity: int, output_len: int, d:int, f_func):
    """
    Keccak Sponge
    msg: 输入字节
    rate: 比特率 (bits)
    capacity: 容量 (bits)
    output_len: 输出长度 (bytes)
    f_func: permutation 函数 (比如 Keccak_f)
    """
    assert (rate + capacity) in [25,50,100,200,400,800,1600]
    block_size = rate // 8

    # 1. Padding
    padded = pad10star1(msg, rate, d)

    # 2. Absorb
    state = Absorb(padded, block_size, f_func)

    #print("== After absorb ==")
    #for x in range(5):
        #for y in range(5):
            #print(f"state[{x}][{y}] is {hex(state[x][y])}")

    # 3. Squeeze
    Z = bytearray()
    #print("output_len is ", output_len)
    while len(Z) < output_len:
        Z.extend(state.T.flatten()[:block_size].tobytes())
        #print("len_Z is ", len(Z))
        if len(Z) >= output_len:
            break
        state = f_func(1600, state)
    # 替换第159行
    #print("Z (hex) is ", Z[:output_len].hex())
#print("Z (bytes) is ", [hex(b) for b in Z[:output_len]])

    return bytes(Z[:output_len])



def SHA3_256(msg: bytes):
    rate = 1088
    capacity = 1600 - rate
    output_len = 32
    d = 0x06 # FOR SHA3 Useage only
    return keccak_sponge(msg, rate, capacity, output_len,d, Keccak_f)

def keccak_256(msg: bytes):
    rate = 1088
    capacity = 1600 - rate
    output_len = 32
    d = 0x01 # FOR Keccak Useage only
    return keccak_sponge(msg, rate, capacity, output_len,d, Keccak_f)

def keccak_512(msg: bytes):
    rate = 576
    capacity = 1600 - rate
    output_len = 64
    d = 0x01 # FOR Keccak Useage only
    return keccak_sponge(msg, rate, capacity, output_len,d, Keccak_f)