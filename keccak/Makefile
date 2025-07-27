
setup:
	python3 setup.py build_ext --inplace

keccak_test:
	@echo "正在编译 keccak 测试..."
	nvcc -o keccak_test keccak_test.cu -lcuda 
	./keccak_test	


keccak_256_test:
	@echo "正在编译 keccak_256 测试..."
	nvcc -o test_keccak_256 test_keccak_256.cu -lcuda 
	./test_keccak_256

clean:
	rm -f test_keccak_256 keccak_test

.PHONY: setup keccak_256_test clean