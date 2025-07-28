import random
import numpy as np
import sympy as sp


def mle_polynomial(x_vars, vec):
    """
    计算多线性扩展多项式在点 x_vars 处的值
    x_vars: 长度为 bits_num 的数组，表示变量 x_0, x_1, ..., x_{bits_num-1}
    vec: 实际需要做MLE的数组
    """
    result = 0
     
    # 遍历所有可能的二进制索引
    for i in range(len(vec)):
        # 将索引 i 转换为二进制数组
        binary_repr = [(i >> j) & 1 for j in range(bits_num)]
        
        # 计算基函数 ∏(x_j^{b_j} * (1-x_j)^{1-b_j})
        basis_value = 1
        for j in range(bits_num):
            if binary_repr[j] == 1:
                basis_value *= x_vars[j]
            else:
                basis_value *= (1 - x_vars[j])
        
        # 累加到结果中
        result += vec[i] * basis_value
     
    return result

def vec_padding(vec: np.array):
    '''
    input vector
    output MLE circuit
    '''
    # 计算输入向量长度的以2为底的对数
    bits_num = int(np.ceil(np.log2(len(vec))))

    padded_size = 2 ** bits_num
    if len(vec) < padded_size:
        padded_vec = np.zeros(padded_size)
        padded_vec[:(len(vec))] = vec
        return padded_vec
    else:
        return vec


def mle_polynomial_symbolic(vec_in, var_names=None):
    """
    将向量转换为符号化的多线性扩展多项式
    vec: 输入向量
    var_names: 变量名列表，如 ['x0', 'x1', 'x2']
    返回: sympy 多项式表达式
    """

    vec = vec_padding(vec_in)

    bits_num = int(np.ceil(np.log2(len(vec))))
    
    # 创建符号变量
    if var_names is None:
        var_names = [f'x{i}' for i in range(bits_num)]
    
    x_vars = [sp.Symbol(name) for name in var_names]
    
    # 构建多项式
    polynomial = 0
    
    for i in range(len(vec)):
        if vec[i] == 0:  # 跳过系数为0的项
            continue
            
        # 将索引 i 转换为二进制表示
        binary_repr = [(i >> j) & 1 for j in range(bits_num)]
        
        # 构建基函数项
        term = vec[i]  # 系数
        for j in range(bits_num):
            if binary_repr[j] == 1:
                term *= x_vars[j]
            else:
                term *= (1 - x_vars[j])
        
        polynomial += term
    
    return sp.expand(polynomial)



    



