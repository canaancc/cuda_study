from sumcheck import sumcheck_protocol
import numpy as np



def test_sumcheck():
    """
    测试sumcheck协议
    """
    print("=== Sumcheck协议测试 ===")
    
    # 测试用例1: 简单向量
    print("\n测试用例1: [1, 2, 3, 4]")
    vec1 = np.array([1, 2, 3, 4])
    success1, msg1 = sumcheck_protocol(vec1)
    print(f"结果: {msg1}")
    
    # 测试用例2: 更复杂的向量
    print("\n测试用例2: [2, 5, 7, 8, 10, 0, 9, 3]")
    vec2 = np.array([2, 5, 7, 8, 10, 0, 9, 3])
    success2, msg2 = sumcheck_protocol(vec2)
    print(f"结果: {msg2}")
    
    # 验证正确性：直接计算向量元素和
    # 理论上 : MLE多项式在所有布尔点 {0,1}^n 上的求和应该等于原始向量的元素和
    print("\n=== 正确性验证 ===")
    print(f"向量1直接求和: {sum(vec1)}")
    print(f"向量2直接求和: {sum(vec2)}")


if __name__ == "__main__":
    test_sumcheck()