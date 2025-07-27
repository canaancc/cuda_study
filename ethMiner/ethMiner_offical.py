from keccak.keccak import keccak_256
import operator
import copy
import time

WORD_BYTES = 4                    # bytes in word
# 原始参数太大，调整为测试友好的小参数
DATASET_BYTES_INIT = 2**12        # 64KB (原: 1GB) - 减少1万倍
DATASET_BYTES_GROWTH = 2**8      # 1KB (原: 8MB) - 大幅减少增长
CACHE_BYTES_INIT = 2**10          # 4KB (原: 16MB) - 减少4000倍
CACHE_BYTES_GROWTH = 2**6         # 256B (原: 128KB) - 大幅减少增长
CACHE_MULTIPLIER=1024             # Size of the DAG relative to the cache
EPOCH_LENGTH = 30000              # blocks per epoch (保持不变)
MIX_BYTES = 128                   # width of mix (保持不变)
HASH_BYTES = 64                   # hash length in bytes (保持不变)
DATASET_PARENTS = 16              # 16 (原: 256) - 减少计算复杂度
CACHE_ROUNDS = 2                  # 2 (原: 3) - 减少cache生成轮数
ACCESSES = 8                      # 8 (原: 64) - 减少hashimoto访问次数


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
    # 计算需要的字节数，最少64字节以匹配keccak_256的输出
    try:
        return value.to_bytes(32, 'big')
    except OverflowError:
        # 如果值太大，使用模运算确保在64字节范围内
        return (value % (2**256)).to_bytes(32, 'big')

def zpad(data, length):
    """零填充到指定长度"""
    if len(data) >= length:
        return data[:length]
    return data + b'\x00' * (length - len(data))

import threading
import queue
from random import randint

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
            seed = keccak_256(seed)
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
        o = [keccak_256(self.seed)]
        for i in range(1, n):
            o.append(keccak_256(o[-1]))

        # Use a low-round version of randmemohash
        for j in range(CACHE_ROUNDS):
            print("Miner start making Cache round {}".format(j))
            for i in range(n):
                #if(i % 10000 == 0):
                   # print("Miner finish making Cache round {} item {}".format(j, i))
                v = o[i][0] % n
                o[i] = keccak_256(list(map(xor, o[(i-1+n) % n], o[v])))

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
        
        # 将mix转换回bytes进行keccak_256计算
        mix_bytes = b''.join(x.to_bytes(4, 'little') for x in mix)
        mix_hash = keccak_256(mix_bytes)
        
        # 将keccak结果转换为整数列表用于后续计算
        if isinstance(mix_hash, bytes):
            mix = [int.from_bytes(mix_hash[j:j+4], 'little') for j in range(0, len(mix_hash), 4)]
        else:
            mix = list(mix_hash)
        
        # fnv it with a lot of random cache nodes based on i
        for j in range(DATASET_PARENTS):
            # Fix: Use modulo len(mix) to prevent index out of range
            cache_index = self.fnv(i ^ j, mix[j % len(mix)])
            cache_item = self.cache[cache_index % n]
            
            # 确保cache_item也是整数列表
            if isinstance(cache_item, bytes):
                cache_data = [int.from_bytes(cache_item[k:k+4], 'little') for k in range(0, len(cache_item), 4)]
            else:
                cache_data = list(cache_item)
                
            mix = list(map(self.fnv, mix, cache_data))
        
        # 最终结果转换为bytes进行keccak_512
        final_mix_bytes = b''.join(x.to_bytes(4, 'little') for x in mix)
        return keccak_256(final_mix_bytes)

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
        # 处理nonce，支持整数和bytes两种输入
        if isinstance(nonce, bytes):
            # 如果nonce已经是bytes，直接使用
            nonce_bytes = nonce
        else:
            # 如果nonce是整数，转换为8字节的bytes对象
            nonce_bytes = nonce.to_bytes(8, 'little')
        
        s = keccak_256(header + nonce_bytes[::-1])
        
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
            "result": serialize_hash(keccak_256(s + cmix_bytes))
        }

    def hashimoto_light(self, header, nonce):
        return self.hashimoto(header, nonce, lambda x: self.calc_dataset_item(x))

    def hashimoto_full(self, header, nonce):
        return self.hashimoto(header, nonce, lambda x: self.dataset[x])

    def mine(self, header, difficulty, num_threads=1):
        """多线程挖矿函数
        
        Args:
            header: 区块头
            difficulty: 挖矿难度
            num_threads: 线程数量，默认为4
        
        Returns:
            找到的nonce值
        """
        # 计算目标值
        target_value = 2**256 // difficulty
        print(f"target_value: {target_value}")
        target = zpad(encode_int(target_value), 32)[::-1]
        
        # 用于存储结果的队列
        result_queue = queue.Queue()
        # 停止标志
        stop_event = threading.Event()
        
        def worker_thread(thread_id, start_nonce, step):
            """工作线程函数"""
            print(f">>> 线程 {thread_id} 开始工作，起始nonce: {start_nonce}")
            nonce = start_nonce
            count = 0
            while not stop_event.is_set():
                try:
                    # 每1000次计算打印一次进度
                    if count % 1000 == 0:
                        print(f"线程 {thread_id} 进度: nonce={nonce}, 计算次数={count}")
                    
                    # 检查当前nonce是否满足难度要求
                    hash_result = self.hashimoto_full(header, nonce)
                    if  int.from_bytes(hash_result["result"], 'big') <= target_value:
                        print(f"线程 {thread_id} 找到有效nonce: {nonce}")
                        result_queue.put(nonce)
                        stop_event.set()  # 通知其他线程停止
                        return
                    
                    # 递增nonce
                    nonce = (nonce + step) % (2**64)
                    count += 1
                    
                    # 防止nonce溢出回到起始值
                    if nonce == start_nonce:
                        print(f"线程 {thread_id} nonce范围遍历完成")
                        break
                        
                except Exception as e:
                    print(f"线程 {thread_id} 发生错误: {e}")
                    break
        
        # 创建并启动工作线程
        threads = []
        for i in range(num_threads):
            # 为每个线程分配不同的起始nonce和步长
            start_nonce = i * (2**64 // num_threads)
            thread = threading.Thread(
                target=worker_thread, 
                args=(i, start_nonce, num_threads)
            )
            thread.daemon = True  # 设置为守护线程
            threads.append(thread)
            thread.start()
            print(f"启动线程 {i}，起始nonce: {start_nonce}")
        
        # 等待结果
        try:
            # 等待任一线程找到结果，超时时间为3000秒
            result_nonce = result_queue.get(timeout=3000)
            stop_event.set()  # 确保所有线程停止
            
            # 等待所有线程结束
            for thread in threads:
                thread.join(timeout=1)
            
            print(f"挖矿成功！找到nonce: {result_nonce}")
            return result_nonce
            
        except queue.Empty:
            print("挖矿超时，未找到有效nonce")
            stop_event.set()
            
            # 等待所有线程结束
            for thread in threads:
                thread.join(timeout=1)
            
            return None

