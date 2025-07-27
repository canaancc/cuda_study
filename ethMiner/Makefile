# Makefile for Eth Miner
NVCC = nvcc
CUDA_FLAGS = -arch=sm_60 -O2
INCLUDE_DIRS = -I../keccak
LIBS = -lcudart

# 目标文件
TARGET = ethMiner
SOURCES = ethMiner_host.cpp ethMiner.cu ../keccak/keccak_256.cu
OBJECTS = ethMiner_host.o ethMiner.o

# 默认目标
all: $(TARGET)

# 编译主程序
$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $^ $(LIBS)

# 编译主机代码
ethMiner_host.o: ethMiner_host.cpp
	$(NVCC) -c $< -o $@ $(INCLUDE_DIRS)

# 编译 CUDA kernel
ethMiner.o: ethMiner.cu
	$(NVCC) -c $< -o $@ $(CUDA_FLAGS) $(INCLUDE_DIRS)

# 编译 keccak 库
# 修改目标文件列表，移除 keccak_256.o
OBJECTS = ethMiner_host.o ethMiner.o

# 移除或注释掉 keccak_256.o 的编译规则
# keccak_256.o: ../keccak/keccak_256.cu
# 	$(NVCC) -c $< -o $@ $(CUDA_FLAGS)

# 清理
clean:
	rm -f $(OBJECTS) $(TARGET) *.o

# 运行
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run