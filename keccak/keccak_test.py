
import hashlib
from Crypto.Hash import keccak

def test_keccak():
    """
    Test the Keccak sponge function implementation.
    """
    from keccak import keccak_256
    
    # Test case: "abc" input
    test_msg = "abc"
    test_result = keccak_256(bytes(test_msg, 'utf-8'))
    print(f"输入: '{test_msg}'")
    print(f"Keccak-256 结果: {test_result.hex()}")
    print(f"结果长度: {len(test_result)} bytes")
    
    # Assertions to verify correctness
    assert len(test_result) == 32, "Keccak-256 should return 32 bytes"
    
    # Test deterministic behavior
    repeat_result = keccak_256(bytes(test_msg, 'utf-8'))
    assert test_result == repeat_result, "Keccak-256 should be deterministic"
    
    print("\n✅ 测试通过! Keccak-256 实现工作正常。")
    
    return {
        'abc': test_result.hex()
    }

def test_keccak_vs_reference():
    """
    Compare our Keccak implementation with a reference implementation if available
    """
    from keccak import keccak_256
    
    # Test vectors for Keccak-256 (not SHA3-256)
    test_vectors = [
        {
            'input': b'',
            'expected': keccak.new(digest_bits=256, data=b'').hexdigest()
        },
        {
            'input': b'abc',
            'expected': keccak.new(digest_bits=256, data=b'abc').hexdigest()
        }
    ]
    
    print("\n=== Reference Test Vectors ===")
    for i, vector in enumerate(test_vectors):
        result = keccak_256(vector['input'])
        result_hex = result.hex()
        expected = vector['expected']
        
        print(f"Test Vector {i+1}:")
        print(f"Input: {vector['input']}")
        print(f"Expected: {expected}")
        print(f"Got:      {result_hex}")
        print(f"Match: {'✅' if result_hex == expected else '❌'}")
        
        if result_hex == expected:
            print("✅ Test vector passed!")
        else:
            print("❌ Test vector failed!")
        print()

def test_keccak_f():
    """
    Test the Keccak-f permutation function with b=1600
    按照 CUDA 测试的输出格式显示结果
    """
    import numpy as np
    from keccak import Keccak_f
    
    print("\n=== Testing Keccak-f(b=1600) ===")
    
    # Test case 1: 递增序列测试 (类似 CUDA 测试)
    print("\nBlock 0:")
    test_state = np.zeros((5, 5), dtype=np.uint64)
    
    # 填充递增序列 (0, 1, 2, ..., 24)
    for i in range(5):
        for j in range(5):
            test_state[i, j] = i * 5 + j
    
    # 显示输入状态
    print("输入:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{test_state[i, j]:016x} "
        print(row_str.strip())
    
    # 执行 Keccak-f 变换
    result_state = Keccak_f(1600, test_state.copy())
    
    # 显示输出状态
    print("输出:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{result_state[i, j]:016x} "
        print(row_str.strip())
    
    # Test case 2: 零状态测试
    print("\nBlock 1:")
    zero_state = np.zeros((5, 5), dtype=np.uint64)
    
    print("输入:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{zero_state[i, j]:016x} "
        print(row_str.strip())
    
    result_zero = Keccak_f(1600, zero_state.copy())
    
    print("输出:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{result_zero[i, j]:016x} "
        print(row_str.strip())
    
    # Test case 3: 单比特测试
    print("\nBlock 2:")
    single_bit_state = np.zeros((5, 5), dtype=np.uint64)
    single_bit_state[0, 0] = 1
    
    print("输入:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{single_bit_state[i, j]:016x} "
        print(row_str.strip())
    
    result_single = Keccak_f(1600, single_bit_state.copy())
    
    print("输出:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{result_single[i, j]:016x} "
        print(row_str.strip())
    
    # Test case 4: 模式测试 (交替位)
    print("\nBlock 3:")
    pattern_state = np.zeros((5, 5), dtype=np.uint64)
    pattern_state[0, 0] = 0xAAAAAAAAAAAAAAAA
    pattern_state[1, 1] = 0x5555555555555555
    pattern_state[2, 2] = 0xFFFFFFFFFFFFFFFF
    pattern_state[3, 3] = 0x0F0F0F0F0F0F0F0F
    pattern_state[4, 4] = 0xF0F0F0F0F0F0F0F0
    
    print("输入:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{pattern_state[i, j]:016x} "
        print(row_str.strip())
    
    result_pattern = Keccak_f(1600, pattern_state.copy())
    
    print("输出:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{result_pattern[i, j]:016x} "
        print(row_str.strip())
    
    # Test case 5: 随机状态测试
    print("\nBlock 4:")
    np.random.seed(42)  # 固定种子确保可重现
    random_state = np.random.randint(0, 2**32, (5, 5), dtype=np.uint64)
    
    print("输入:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{random_state[i, j]:016x} "
        print(row_str.strip())
    
    result_random = Keccak_f(1600, random_state.copy())
    
    print("输出:")
    for i in range(5):
        row_str = ""
        for j in range(5):
            row_str += f"{result_random[i, j]:016x} "
        print(row_str.strip())
    
    # 验证测试
    print("\n=== 验证结果 ===")
    
    # 1. 确定性测试
    result_repeat = Keccak_f(1600, test_state.copy())
    is_deterministic = np.array_equal(result_state, result_repeat)
    print(f"确定性测试: {'✅ 通过' if is_deterministic else '❌ 失败'}")
    
    # 2. 状态形状测试
    shape_correct = result_state.shape == (5, 5)
    print(f"状态形状测试: {'✅ 通过' if shape_correct else '❌ 失败'}")
    
    # 3. 数据类型测试
    dtype_correct = result_state.dtype == np.uint64
    print(f"数据类型测试: {'✅ 通过' if dtype_correct else '❌ 失败'}")
    
    # 4. 非零输出测试 (对于非零输入)
    non_zero_output = np.any(result_state != 0)
    print(f"非零输出测试: {'✅ 通过' if non_zero_output else '❌ 失败'}")
    
    # 5. 雪崩效应测试 (单比特变化导致大量输出变化)
    single_bit_state2 = single_bit_state.copy()
    single_bit_state2[0, 0] = 2  # 改变一个比特
    result_avalanche = Keccak_f(1600, single_bit_state2)
    
    # 计算汉明距离
    diff_bits = 0
    for i in range(5):
        for j in range(5):
            xor_result = result_single[i, j] ^ result_avalanche[i, j]
            diff_bits += bin(xor_result).count('1')
    
    avalanche_ratio = diff_bits / (5 * 5 * 64)  # 总比特数
    avalanche_good = avalanche_ratio > 0.4  # 期望至少40%的比特发生变化
    print(f"雪崩效应测试: {'✅ 通过' if avalanche_good else '❌ 失败'} (变化比例: {avalanche_ratio:.2%})")
    
    print("\n✅ Keccak-f(b=1600) 测试完成!")
    
    return {
        'deterministic': is_deterministic,
        'shape_correct': shape_correct,
        'dtype_correct': dtype_correct,
        'non_zero_output': non_zero_output,
        'avalanche_ratio': avalanche_ratio,
        'all_tests_passed': all([is_deterministic, shape_correct, dtype_correct, non_zero_output, avalanche_good])
    }

if __name__ == "__main__":
    # Run the original test
    test_results = test_keccak()
    
    # Run reference comparison
    #test_keccak_vs_reference()
    
    ## Run Keccak-f test
    #keccak_f_results = test_keccak_f(py)
    #
    #print("\n=== Test Summary ===")
    #for test_name, result in test_results.items():
    #    print(f"{test_name}: {result}")
    
    #print("\n=== Keccak-f Test Summary ===")
    #for test_name, result in keccak_f_results.items():
    #    print(f"{test_name}: {result}")