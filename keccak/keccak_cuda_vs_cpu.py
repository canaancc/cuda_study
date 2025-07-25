import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
from keccak import Keccak_f


# 1. 读取并编译你的 CUDA 源码
with open("keccak.cu", "r") as f:
    mod = SourceModule(f.read())

# 2. 获取你的 CUDA 函数
keccak_kernel = mod.get_function("keccak_kernel")

# 3. 准备输入数据
blocks = 1_000_000  # 例如 1M 个 Keccak 状态
host_input = np.random.randint(0, 2**64, size=(blocks, 25), dtype=np.uint64)
host_output = np.zeros_like(host_input)

# 4. 分配 GPU 内存并拷贝数据
d_input = cuda.mem_alloc(host_input.nbytes)
d_output = cuda.mem_alloc(host_output.nbytes)
cuda.memcpy_htod(d_input, host_input)


# 5. 分配线程块和线程网格
thread_per_block = 256
grid_dim = (blocks + thread_per_block - 1) // thread_per_block


# 6. 执行 CUDA 函数
cuda_start_time = time.time()
keccak_kernel(d_input, d_output, np.uint32(blocks), block=(thread_per_block, 1, 1), grid=(grid_dim, 1))
cuda.Context.synchronize()
cuda_end_time = time.time()
print(f"CUDA Time: {cuda_end_time - cuda_start_time:.6f} seconds")


# 7. 拷贝结果回主机
copy_start = time.time()
cuda.memcpy_dtoh(host_output, d_output)
copy_end = time.time()
print(f"Copy Time: {copy_end - copy_start:.6f} seconds")


############ Python Test ##################

python_out = np.zeros_like(host_input)
state_in = np.zeros((5, 5), dtype=np.uint64)
state_out = np.zeros((5, 5), dtype=np.uint64)
python_start_time = time.time()
for i in range(100):
    for x in range(5):
        for y in range(5):
            state_in[x][y] = host_input[i][x+y*5]
    state_out = Keccak_f(1600, state_in)
    for x in range(5):
        for y in range(5):
            python_out[i][x+y*5] = state_out[x][y]
    
python_end_time = time.time()
print(f"Python Time: {python_end_time - python_start_time:.6f} seconds")





