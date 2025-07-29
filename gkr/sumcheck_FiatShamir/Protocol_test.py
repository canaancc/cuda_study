from .Protocol import sumcheck_for_vector
import numpy as np


def test_basic():
    vec = [1,2,3,4,5,6,7,8]
    result = sumcheck_for_vector(vec)
    print("测试结果:")
    print(result)
    return result 

def test_protocol():
    """测试协议功能"""
    vec = np.random.randint(1, 1000, 2**8)  # 生成100个随机浮点数组成的向量
    result = sumcheck_for_vector(vec)
    print("测试结果:")
    print(result)
    return result

if __name__ == "__main__":
    test_protocol()
