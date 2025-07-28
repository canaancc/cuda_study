from ast import main
import numpy as np
from mle import mle_polynomial_symbolic
import sympy as sp

def test_mle():
    # 使用示例
    vec = np.array([2,5,7,8,10,0,9])
    poly = mle_polynomial_symbolic(vec)
    
    print(f"MLE 多项式: {poly}")
    
    # 方法1：输入具体的 x0, x1, x2 值进行求解
    x0_val = 0.5
    x1_val = 0.3
    x2_val = 0.8
    
    # 创建变量替换字典
    x0, x1, x2 = sp.symbols('x0 x1 x2')
    substitutions = {x0: x0_val, x1: x1_val, x2: x2_val}
    
    # 计算多项式在给定点的值
    result = poly.subs(substitutions)
    print(f"当 x0={x0_val}, x1={x1_val}, x2={x2_val} 时，多项式值为: {result}")
    
    # 方法2：使用 lambdify 创建数值函数（更高效）
    poly_func = sp.lambdify([x0, x1, x2], poly, 'numpy')
    result_fast = poly_func(x0_val, x1_val, x2_val)
    print(f"使用 lambdify 计算结果: {result_fast}")

def test_mle_with_input():
    """交互式输入 x 值进行求解"""
    vec = np.array([2,5,7,8,10,0,9])
    poly = mle_polynomial_symbolic(vec)
    
    print(f"MLE 多项式: {poly}")
    
    # 获取多项式中的变量
    variables = list(poly.free_symbols)
    variables.sort(key=lambda x: x.name)  # 按变量名排序
    
    print(f"\n该多项式包含变量: {[str(var) for var in variables]}")
    
    # 输入各变量的值
    substitutions = {}
    for var in variables:
        value = float(input(f"请输入 {var} 的值: "))
        substitutions[var] = value
    
    # 计算结果
    result = poly.subs(substitutions)
    print(f"\n计算结果: {result}")
    print(f"数值结果: {float(result)}")

def test_multiple_points():
    """测试多个点的值"""
    vec = np.array([2,5,7,8,10,0,9])
    print("Array = ", vec)

    poly = mle_polynomial_symbolic(vec)
    
    print(f"MLE 多项式: {poly}")
    
    # 创建符号变量
    x0, x1, x2 = sp.symbols('x0 x1 x2')
    
    # 测试多个点
    test_points = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
        (0.5, 0.5, 0.5)
    ]
    
    print("\n测试多个点的值:")
    for point in test_points:
        substitutions = {x0: point[0], x1: point[1], x2: point[2]}
        result = poly.subs(substitutions)
        print(f"({point[0]}, {point[1]}, {point[2]}) -> {result}")

if __name__ == "__main__":
    print("=== 基本测试 ===")
    test_mle()
    
    print("\n=== 交互式输入测试 ===")
    # test_mle_with_input()  # 取消注释以启用交互式输入
    
    print("\n=== 多点测试 ===")
    test_multiple_points()
