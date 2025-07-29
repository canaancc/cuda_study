from .Protocol import sumcheck_for_vector

def test_protocol():
    """测试协议功能"""
    vec = [1, 2, 3, 4]
    result = sumcheck_for_vector(vec)
    print("测试结果:")
    print(result)
    return result

if __name__ == "__main__":
    test_protocol()
