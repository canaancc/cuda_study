#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EthMiner 挖矿功能测试用例
测试 ethMiner 类的各种功能，包括 DAG 生成、哈希计算和挖矿过程
"""

import sys
import os
import time
import unittest
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ethMiner.ethMiner_offical import ethMiner, serialize_hash, encode_int, zpad

class TestEthMiner(unittest.TestCase):
    """ethMiner 测试类"""
    
    def setUp(self):
        """测试前准备"""
        print(f"\n{'='*50}")
        print(f"开始测试: {self._testMethodName}")
        print(f"{'='*50}")
        
        # 使用较小的区块号以减少计算时间
        self.test_block_number = 0  # 创世区块
        self.test_header = b'test_header_12345678901234569870123456789012'  # 32字节
        self.test_difficulty = 1000  # 较低难度便于测试
        
    def tearDown(self):
        """测试后清理"""
        print(f"测试完成: {self._testMethodName}")
        print(f"{'='*50}\n")

    def test_miner_initialization(self):
        """测试矿工初始化"""
        print("测试矿工初始化...")
        
        start_time = time.time()
        miner = ethMiner(self.test_block_number)
        init_time = time.time() - start_time
        
        # 验证基本属性
        self.assertEqual(miner.epoch_number, 0)
        self.assertGreater(miner.cache_size, 0)
        self.assertGreater(miner.full_size, 0)
        self.assertIsNotNone(miner.seed)
        self.assertIsNotNone(miner.cache)
        self.assertIsNotNone(miner.dataset)
        
        print(f"✅ 初始化成功")
        print(f"   - Epoch: {miner.epoch_number}")
        print(f"   - Cache Size: {miner.cache_size:,} bytes")
        print(f"   - Dataset Size: {miner.full_size:,} bytes")
        print(f"   - Cache Items: {len(miner.cache):,}")
        print(f"   - Dataset Items: {len(miner.dataset):,}")
        print(f"   - 初始化时间: {init_time:.2f} 秒")

    def test_hashimoto_functions(self):
        """测试 hashimoto 相关函数"""
        print("测试 hashimoto 函数...")
        
        miner = ethMiner(self.test_block_number)
        test_nonce = (12345).to_bytes(8, 'little')
        
        # 测试 hashimoto_light
        print("测试 hashimoto_light...")
        start_time = time.time()
        result_light = miner.hashimoto_light(self.test_header, test_nonce)
        light_time = time.time() - start_time
        
        self.assertIn("mix digest", result_light)
        self.assertIn("result", result_light)
        self.assertIsInstance(result_light["mix digest"], bytes)
        self.assertIsInstance(result_light["result"], bytes)
        
        # 测试 hashimoto_full
        print("测试 hashimoto_full...")
        start_time = time.time()
        result_full = miner.hashimoto_full(self.test_header, test_nonce)
        full_time = time.time() - start_time
        
        self.assertIn("mix digest", result_full)
        self.assertIn("result", result_full)
        self.assertIsInstance(result_full["mix digest"], bytes)
        self.assertIsInstance(result_full["result"], bytes)
        
        # 验证两种方法结果一致
        self.assertEqual(result_light["result"], result_full["result"])
        self.assertEqual(result_light["mix digest"], result_full["mix digest"])
        
        print(f"✅ Hashimoto 测试通过")
        print(f"   - Light 模式时间: {light_time:.4f} 秒")
        print(f"   - Full 模式时间: {full_time:.4f} 秒")
        print(f"   - 性能提升: {light_time/full_time:.1f}x")
        print(f"   - 结果哈希: {result_full['result'][:8].hex()}...")

    def test_mining_process(self):
        """测试挖矿过程"""
        print("测试挖矿过程...")
        
        miner = ethMiner(self.test_block_number)
        
        # 使用较高难度确保能找到解
        test_difficulty = 10000
        
        print(f"开始挖矿 (难度: {test_difficulty})...")
        start_time = time.time()
        
        # 模拟挖矿过程，限制最大尝试次数
        with patch('random.randint', return_value=0):  # 固定起始nonce
            result_nonce = miner.mine(self.test_header, test_difficulty)
        
        mining_time = time.time() - start_time
        
        # 验证挖矿结果
        self.assertIsInstance(result_nonce, int)
        self.assertGreaterEqual(result_nonce, 0)
        self.assertLess(result_nonce, 2**64)
        
        # 验证找到的nonce确实满足难度要求
        nonce_bytes = result_nonce.to_bytes(8, 'little')
        hash_result = miner.hashimoto_full(self.test_header, nonce_bytes)
        target = 2**256 // test_difficulty
        
        # Convert bytes result to int for comparison
        result_int = int.from_bytes(hash_result["result"], 'big')
        self.assertLessEqual(result_int, target)
        
        print(f"✅ 挖矿成功")
        print(f"   - 找到的 Nonce: {result_nonce}")
        print(f"   - 挖矿时间: {mining_time:.4f} 秒")
        print(f"   - 结果哈希: {hash_result['result'][:8].hex()}...")
        print(f"   - 目标值: {hex(target)[:10]}...")

    def test_utility_functions(self):
        """测试工具函数"""
        print("测试工具函数...")
        
        # 测试 serialize_hash
        test_list = [0x12345678, 0x9abcdef0]
        serialized = serialize_hash(test_list)
        self.assertIsInstance(serialized, bytes)
        self.assertEqual(len(serialized), 8)  # 2 * 4 bytes
        
        # 测试 encode_int
        test_int = 12345
        encoded = encode_int(test_int)
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(len(encoded), 32)
        
        # 测试 zpad
        test_data = b'hello'
        padded = zpad(test_data, 10)
        self.assertEqual(len(padded), 10)
        self.assertTrue(padded.startswith(b'hello'))
        
        print(f"✅ 工具函数测试通过")
        print(f"   - serialize_hash: {serialized.hex()}")
        print(f"   - encode_int: {encoded[:8].hex()}...")
        print(f"   - zpad: {padded}")

    def test_performance_comparison(self):
        """测试性能对比"""
        print("测试性能对比...")
        
        miner = ethMiner(self.test_block_number)
        test_nonce = (54321).to_bytes(8, 'little')
        
        # 多次测试取平均值
        iterations = 5
        light_times = []
        full_times = []
        
        for i in range(iterations):
            # 测试 light 模式
            start = time.time()
            miner.hashimoto_light(self.test_header, test_nonce)
            light_times.append(time.time() - start)
            
            # 测试 full 模式
            start = time.time()
            miner.hashimoto_full(self.test_header, test_nonce)
            full_times.append(time.time() - start)
        
        avg_light = sum(light_times) / len(light_times)
        avg_full = sum(full_times) / len(full_times)
        speedup = avg_light / avg_full
        
        print(f"✅ 性能对比完成 ({iterations} 次平均)")
        print(f"   - Light 模式平均时间: {avg_light:.6f} 秒")
        print(f"   - Full 模式平均时间: {avg_full:.6f} 秒")
        print(f"   - Full 模式加速比: {speedup:.1f}x")
        
        # Full 模式应该明显更快
        self.assertGreater(speedup, 1.0)

    def test_different_difficulties(self):
        """测试不同难度下的挖矿"""
        print("测试不同难度...")
        
        miner = ethMiner(self.test_block_number)
        difficulties = [1, 5, 10]
        
        for difficulty in difficulties:
            print(f"\n测试难度: {difficulty}")
            
            # 使用固定的随机种子确保可重现性
            with patch('random.randint', return_value=0):
                print("=== Start Mining in difficulty at ", difficulty, " ===")
                start_time = time.time()
                nonce = miner.mine(self.test_header, difficulty)
                mining_time = time.time() - start_time
            
            # 验证结果
            nonce_bytes = nonce.to_bytes(8, 'little')
            result = miner.hashimoto_full(self.test_header, nonce_bytes)
            target = zpad(encode_int(2**256 // difficulty), 64)[::-1]
            
            self.assertLessEqual(result["result"], target)
            
            print(f"   - Nonce: {nonce}")
            print(f"   - 时间: {mining_time:.4f} 秒")
            print(f"   - 哈希: {result['result'][:8].hex()}...")

def run_mining_tests():
    """运行所有挖矿测试"""
    print("\n" + "="*60)
    print("🚀 开始 EthMiner 挖矿功能测试")
    print("="*60)
    
    # 创建测试套件
    #test_suite = unittest.TestLoader().loadTestsFromTestCase(TestEthMiner)
    test_suite = unittest.TestSuite()

    test_suite.addTest(TestEthMiner('test_mining_process'))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
        print(f"失败: {len(result.failures)}")
        print(f"错误: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # 运行测试
    success = run_mining_tests()
    
    # 退出码
    sys.exit(0 if success else 1)