from keccak.keccak import keccak_512
import operator
import copy
import time

WORD_BYTES = 4                    # bytes in word
# 原始参数太大，调整为测试友好的小参数
DATASET_BYTES_INIT = 2**16        # 64KB (原: 1GB) - 减少1万倍
DATASET_BYTES_GROWTH = 2**10      # 1KB (原: 8MB) - 大幅减少增长
CACHE_BYTES_INIT = 2**12          # 4KB (原: 16MB) - 减少4000倍
CACHE_BYTES_GROWTH = 2**8         # 256B (原: 128KB) - 大幅减少增长
CACHE_MULTIPLIER=1024             # Size of the DAG relative to the cache
EPOCH_LENGTH = 30000              # blocks per epoch (保持不变)
MIX_BYTES = 128                   # width of mix (保持不变)
HASH_BYTES = 64                   # hash length in bytes (保持不变)
DATASET_PARENTS = 16              # 16 (原: 256) - 减少计算复杂度
CACHE_ROUNDS = 2                  # 2 (原: 3) - 减少cache生成轮数
ACCESSES = 8                      # 8 (原: 64) - 减少hashimoto访问次数
DATASET_PARENTS = 256             # number of parents of each dataset element
CACHE_ROUNDS = 3                  # number of rounds in cache production
ACCESSES = 64                     # number of accesses in hashimoto loop

def xor(a, b):
    return a ^ b

def isprime(x):
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
         if x % i == 0:
             return False
    return True

# 添加缺失的函数
def serialize_hash(hash_data):
    """将哈希数据序列化为字节"""
    if isinstance(hash_data, list):
        return b''.join(x.to_bytes(4, 'little') for x in hash_data)
    return hash_data

def encode_int(value):
    """将整数编码为字节"""
    return value.to_bytes(32, 'big')

def zpad(data, length):
    """零填充到指定长度"""
    if len(data) >= length:
        return data[:length]
    return data + b'\x00' * (length - len(data))

class ethMiner():
    def __init__(self,block_number) -> None:
        self.epoch_number = block_number // EPOCH_LENGTH
        self.cache_size = self.get_cache_size()
        self.full_size = self.get_full_size()
        self.seed = self.get_seed()
        self.cache = self.mkcache()
        self.dataset = self.calc_dataset()

    def get_seed(self):
        seed = b'\x00' * 32 # 32bytes
        for _ in range(self.epoch_number):
            seed = keccak_512(seed)
        return seed
 
    def get_cache_size(self):
        sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * self.epoch_number
        sz -= HASH_BYTES
        while not isprime(sz // HASH_BYTES):
            sz -= 2 * HASH_BYTES
        return sz

    def get_full_size(self):
        sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * self.epoch_number
        sz -= MIX_BYTES
        while not isprime(sz // MIX_BYTES):
            sz -= 2 * MIX_BYTES
        return sz

    def mkcache(self):
        print("Miner start making Cache")
        start = time.time()
        n = self.cache_size // HASH_BYTES

        # Sequentially produce the initial dataset
        o = [keccak_512(self.seed)]
        for i in range(1, n):
            o.append(keccak_512(o[-1]))

        # Use a low-round version of randmemohash
        for j in range(CACHE_ROUNDS):
            print("Miner start making Cache round {}".format(j))
            for i in range(n):
                if(i % 10000 == 0):
                    print("Miner finish making Cache round {} item {}".format(j, i))
                v = o[i][0] % n
                o[i] = keccak_512(list(map(xor, o[(i-1+n) % n], o[v])))

        end = time.time()
        print("Miner finish making Cache, time: {:.2f}s".format(end - start))

        return o

    def fnv(self, v1, v2):
        FNV_PRIME = 0x01000193
        return ((v1 * FNV_PRIME) ^ v2) % 2**32

    def calc_dataset_item(self, i):
        n = len(self.cache)
        r = HASH_BYTES // WORD_BYTES
        # initialize the mix - 将bytes转换为整数列表
        cache_item = self.cache[i % n]
        if isinstance(cache_item, bytes):
            # 将bytes转换为32位整数列表
            mix = [int.from_bytes(cache_item[j:j+4], 'little') for j in range(0, len(cache_item), 4)]
        else:
            mix = list(cache_item)  # 如果已经是列表，直接复制
        
        mix[0] ^= i
        
        # 将mix转换回bytes进行keccak_512计算
        mix_bytes = b''.join(x.to_bytes(4, 'little') for x in mix)
        mix_hash = keccak_512(mix_bytes)
        
        # 将keccak结果转换为整数列表用于后续计算
        if isinstance(mix_hash, bytes):
            mix = [int.from_bytes(mix_hash[j:j+4], 'little') for j in range(0, len(mix_hash), 4)]
        else:
            mix = list(mix_hash)
        
        # fnv it with a lot of random cache nodes based on i
        for j in range(DATASET_PARENTS):
            cache_index = self.fnv(i ^ j, mix[j % r])
            cache_item = self.cache[cache_index % n]
            
            # 确保cache_item也是整数列表
            if isinstance(cache_item, bytes):
                cache_data = [int.from_bytes(cache_item[k:k+4], 'little') for k in range(0, len(cache_item), 4)]
            else:
                cache_data = list(cache_item)
                
            mix = list(map(self.fnv, mix, cache_data))
        
        # 最终结果转换为bytes进行keccak_512
        final_mix_bytes = b''.join(x.to_bytes(4, 'little') for x in mix)
        return keccak_512(final_mix_bytes)

    def calc_dataset(self):
        print("Miner start making Dataset")
        start = time.time()
        n = self.full_size // HASH_BYTES
        dataset = []
        for i in range(n):
            dataset.append(self.calc_dataset_item(i))
        end = time.time()
        print("Miner finish making Dataset, time: {:.2f}s".format(end - start))
        return dataset

    def hashimoto(self, header, nonce, dataset_lookup):
        n = self.full_size // HASH_BYTES
        w = MIX_BYTES // WORD_BYTES
        mixhashes = MIX_BYTES // HASH_BYTES
        # combine header+nonce into a 64 byte seed
        # 将nonce转换为8字节的bytes对象
        nonce_bytes = nonce.to_bytes(8, 'little')
        s = keccak_512(header + nonce_bytes[::-1])
        
        # start the mix with replicated s
        # 将s转换为整数列表以便后续计算
        if isinstance(s, bytes):
            s_list = [int.from_bytes(s[j:j+4], 'little') for j in range(0, len(s), 4)]
        else:
            s_list = list(s)
        
        mix = []
        for _ in range(MIX_BYTES // HASH_BYTES):
            mix.extend(s_list)
        
        # mix in random dataset nodes
        for i in range(ACCESSES):
            p = self.fnv(i ^ s_list[0], mix[i % w]) % (n // mixhashes) * mixhashes
            newdata = []
            for j in range(MIX_BYTES // HASH_BYTES):
                dataset_item = dataset_lookup(p + j)
                # 确保dataset_item是整数列表
                if isinstance(dataset_item, bytes):
                    newdata.extend([int.from_bytes(dataset_item[k:k+4], 'little') for k in range(0, len(dataset_item), 4)])
                else:
                    newdata.extend(dataset_item)
            mix = list(map(self.fnv, mix, newdata))
        
        # compress mix
        cmix = []
        for i in range(0, len(mix), 4):
            cmix.append(self.fnv(self.fnv(self.fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]))
        
        # 将cmix转换为bytes以便与s连接
        cmix_bytes = b''.join(x.to_bytes(4, 'little') for x in cmix)
        
        return {
            "mix digest": serialize_hash(cmix),
            "result": serialize_hash(keccak_512(s + cmix_bytes))
        }

    def hashimoto_light(self, header, nonce):
        return self.hashimoto(header, nonce, lambda x: self.calc_dataset_item(x))

    def hashimoto_full(self, header, nonce):
        return self.hashimoto(header, nonce, lambda x: self.dataset[x])

    def mine(self, header, difficulty):
        # zero-pad target to compare with hash on the same digit
        target = zpad(encode_int(2**256 // difficulty), 64)[::-1]
        from random import randint
        nonce = randint(0, 2**64)
        while self.hashimoto_full(header, nonce)["result"] > target:
            nonce = (nonce + 1) % 2**64
        return nonce

