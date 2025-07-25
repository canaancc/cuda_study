from keccak.keccak import keccak_256

### Parameter ###
DATASET_BYTES_INIT = 1073741824  # 1GB初始大小
DATASET_BYTES_GROWTH = 8388608   # 每epoch增长8MB
CACHE_BYTES_INIT = 16777216      # 16MB初始大小
CACHE_BYTES_GROWTH = 131072      # 每epoch增长128KB
EPOCH_LENGTH = 30000             # 每epoch块数
MIX_BYTES = 128                  # mix数据大小
HASH_BYTES = 64                  # 哈希大小
DATASET_PARENTS = 256            # dataset父节点数
CACHE_ROUNDS = 3                 # cache轮数
ACCESSES = 64                    # 访问次数

class ethash_Miner:
    def __init__(self, block_number: int):
        self.EPOCH_NUMBER = block_number // EPOCH_LENGTH
        self.CACH_SIZE = self.gen_cachesize()
        self.SEED = self.generate_seedhash()
        self.CACHE = self.mkcache()
        self.DATASET_SIZE = self.gen_datasize()


    def generate_seedhash(self) -> bytes:
        seed = b'\x00' * 32 # 32bytes
        for _ in range(self.EPOCH_NUMBER):
            seed = keccak_256(seed)
        return seed

    def gen_cachesize(self):
        cache_size = (CACHE_BYTES_INIT + (self.EPOCH_NUMBER * CACHE_BYTES_GROWTH))//1024 * 1024
        return cache_size

def mkcache(cache_size, seed):
    n = cache_size // HASH_BYTES

    # Sequentially produce the initial dataset
    o = [sha3_512(seed)]
    for i in range(1, n):
        o.append(sha3_512(o[-1]))

    # Use a low-round version of randmemohash
    for _ in range(CACHE_ROUNDS):
        for i in range(n):
            v = o[i][0] % n
            o[i] = sha3_512(map(xor, o[(i-1+n) % n], o[v]))

    return o




    def mkcache(self):
        cache = [self.SEED]
        while len(cache) < self.CACH_SIZE :
            cache.append(keccak_256(cache[-1]))
        
        # (2) 混合过程
        for _ in range(3):  # 通常为固定轮次，比如 3 轮
            for i in range(len(cache)):
                selected_index = (cache[i-1] ^ i ) % len(cache)
                cache[i] ^= cache[selected_index]
        return cache


    def gen_datasize(self):
        dataset_size = (DATASET_BYTES_INIT + (self.EPOCH_NUMBER * DATASET_BYTES_GROWTH))//1024 * 1024
        return dataset_size

    def calc_dataset_item(self,i):
        """
        计算dataset中第i个元素
        cache: 已生成的cache数组
        i: dataset中的索引位置
        """
        cache_size = len(self.CACHE)

        # 第一步：初始化mix
        # 使用cache[i % cache_size]和索引i创建初始mix
        initial_hash = cache[i % cache_size]
        mix = keccak_256(initial_hash + int_to_bytes(i, 4))

        # 将64字节扩展到128字节
        mix = mix + mix  # 现在是128字节

        # 第二步：使用256个cache父节点混合
        for j in range(DATASET_PARENTS):
            # 计算要访问的cache索引
            # 使用FNV哈希确保伪随机分布
            cache_index = fnv_hash(i ^ j, mix[j % len(mix)]) % cache_size

            # 获取cache中的数据
            parent_data = cache[cache_index]

            # 使用FNV算法混合数据
            mix = fnv_mix(mix, parent_data)

        # 第三步：最终哈希压缩
        return keccak256(mix)



    
    


