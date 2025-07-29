from .Protocol import sumcheck_for_vector
import numpy as np


def test_basic():
    vec = [1,2,3,4]
    result = sumcheck_for_vector(vec)
    print("测试结果:")
    print(result)
    return result 

def test_protocol():
    """测试协议功能"""
    vec = np.random.randint(1, 2**32-1, 2**4)  # 生成100个随机浮点数组成的向量
    result = sumcheck_for_vector(vec)
    print("测试结果:")
    print(result)
    return result

if __name__ == "__main__":
    test_basic()
