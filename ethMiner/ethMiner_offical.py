from keccak.keccak import keccak_512
import operator
import copy

WORD_BYTES = 4                    # bytes in word
DATASET_BYTES_INIT = 2**30        # bytes in dataset at genesis
DATASET_BYTES_GROWTH = 2**23      # dataset growth per epoch
CACHE_BYTES_INIT = 2**24          # bytes in cache at genesis
CACHE_BYTES_GROWTH = 2**17        # cache growth per epoch
CACHE_MULTIPLIER=1024             # Size of the DAG relative to the cache
EPOCH_LENGTH = 30000              # blocks per epoch
MIX_BYTES = 128                   # width of mix
HASH_BYTES = 64                   # hash length in bytes
DATASET_PARENTS = 256             # number of parents of each dataset element
CACHE_ROUNDS = 3                  # number of rounds in cache production
ACCESSES = 64                     # number of accesses in hashimoto loop

def xor(a, b):
    return a ^ b
def isprime(x):
    for i in range(2, int(x**0.5)):
         if x % i == 0:
             return False
    return True

class ethMiner():
    def __init__(self,block_number, header, nonce) -> None:
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
        while not isprime(sz / HASH_BYTES):
            sz -= 2 * HASH_BYTES
        return sz

    def get_full_size(self):
        sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * self.epoch_number
        sz -= MIX_BYTES
        while not isprime(sz / MIX_BYTES):
            sz -= 2 * MIX_BYTES
        return sz

    def mkcache(self):
        n = self.cache_size // HASH_BYTES

        # Sequentially produce the initial dataset
        o = [keccak_512(self.seed)]
        for i in range(1, n):
            o.append(keccak_512(o[-1]))

        # Use a low-round version of randmemohash
        for _ in range(CACHE_ROUNDS):
            for i in range(n):
                v = o[i][0] % n
                o[i] = keccak_512(map(xor, o[(i-1+n) % n], o[v]))

        return o


    def fnv(self, v1, v2):
        FNV_PRIME = 0x01000193
        return ((v1 * FNV_PRIME) ^ v2) % 2**32

    
    def calc_dataset_item(self, i):
        n = len(self.cache)
        r = HASH_BYTES // WORD_BYTES
        # initialize the mix
        mix = copy.copy(self.cache[i % n])
        mix[0] ^= i
        mix = keccak_512(mix)
        # fnv it with a lot of random cache nodes based on i
        for j in range(DATASET_PARENTS):
            cache_index = self.fnv(i ^ j, mix[j % r])
            mix = map(self.fnv, mix, self.cache[cache_index % n])
        return keccak_512(mix)

    def calc_dataset(self):
        return [self.calc_dataset_item( i) for i in range(self.full_size // HASH_BYTES)]

    def hashimoto(self, header, nonce, dataset_lookup):
        n = self.full_size / HASH_BYTES
        w = MIX_BYTES // WORD_BYTES
        mixhashes = MIX_BYTES / HASH_BYTES
        # combine header+nonce into a 64 byte seed
        s = keccak_512(header + nonce[::-1])
        # start the mix with replicated s
        mix = []
        for _ in range(MIX_BYTES / HASH_BYTES):
            mix.extend(s)
        # mix in random dataset nodes
        for i in range(ACCESSES):
            p = self.fnv(i ^ s[0], mix[i % w]) % (n // mixhashes) * mixhashes
            newdata = []
            for j in range(MIX_BYTES / HASH_BYTES):
                newdata.extend(dataset_lookup(p + j))
            mix = map(self.fnv, mix, newdata)
        # compress mix
        cmix = []
        for i in range(0, len(mix), 4):
            cmix.append(self.fnv(self.fnv(self.fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]))
        return {
            "mix digest": serialize_hash(cmix),
            "result": serialize_hash(keccak_512(s+cmix))
        }

    def hashimoto_light(self, header, nonce):
        return self.hashimoto(header, nonce, self.full_size, lambda x: self.calc_dataset_item(x))

    def hashimoto_full(self, header, nonce):
        return self.hashimoto(header, nonce, self.full_size, lambda x: self.dataset[x])

    def mine(self, header, difficulty):
        # zero-pad target to compare with hash on the same digit
        target = zpad(encode_int(2**256 // difficulty), 64)[::-1]
        from random import randint
        nonce = randint(0, 2**64)
        while self.hashimoto_full(header, nonce) > target:
            nonce = (nonce + 1) % 2**64
        return nonce

