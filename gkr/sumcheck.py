import sympy as sp
from .mle import mle_polynomial_symbolic, vec_padding
import random

class SumcheckProver:

    def __init__(self, polynomial, variables) -> None:
        """
        初始化证明者
        polynomial: sympy多项式表达式
        variables: 变量列表，按顺序排列 [x0, x1, x2, ...]
        """         
        self.polynomial = polynomial
        self.variables = variables
        self.num_len = len(variables)

    def compute_claimed_sum(self):
        """
        计算声称的总和 H = Σ_{x∈{0,1}^n} f(x)
        便利
        """
        total_sum = 0

        for i in range(2**self.num_len):
            # 将 i 转换为二进制
            assignment = {}
            for j in range(self.num_len):
                bit = (i>>j) & 1
                assignment[self.variables[j]]   = bit
            
            # 计算多项式
            value = self.polynomial.subs(assignment)
            total_sum += value

        return total_sum

    def generate_univariate_polynomial(self, round_num, challenge_values):
        """
        生成第round_num轮的单变量多项式
        challenge_values: 前面轮次验证者发送的挑战值, {x0:3, x1:4 , ...}
        每一轮会将challenge values 里的数据进行替换，再求遍历数据的和 
        """

        # 从challenge value中将数据对poly进行替换
        current_poly = self.polynomial
        for i , val in enumerate(challenge_values):
            current_poly = current_poly.subs(self.variables[i],val)

        # 当前轮的变量 r
        current_value = self.variables[round_num]

        # 对剩余的变量进行遍历得到对应的多项式
        remain_vars = self.variables[round_num+1:]
        
        # 如果是最后一轮， 那么直接就反回current poly
        if not remain_vars:
            return current_poly

        # 对所有剩余变量的布尔遍历求和
        univariate_poly = 0
        for i in range(2**len(remain_vars)):
            assignment = {}
            for j in range(len(remain_vars)):
                bit = (i >> j) & 1
                assignment[remain_vars[j]] = bit
            
            partial_poly = current_poly.subs(assignment)
            univariate_poly += partial_poly
        
        return sp.expand(univariate_poly) # 将多项式进行扩展

    def evaluate_at_points(self, polynomial, var, points):
        """
        计算单变量多项式在指定点的多个数值
        """
        return [polynomial.subs(var, point) for point in points]

class SumcheckVerifier:
    """
    Sumcheck协议的验证者(Verifier)
    """
    def __init__(self, num_vars, claimed_sum) -> None:
        """
        初始化验证者
        num_vars: 变量数量
        claimed_sum: 证明者声称的总和
        """
        self.num_vars = num_vars
        self.claimed_sum = claimed_sum
        self.challenge_values = []
        self.last_polynomial_value = 0

    
    def generate_challenge(self):
        return random.uniform(0,1) # 可以进行调整

    def verify_round(self, univariate_poly, var, round_num):
        """
        验证当前轮次
        univariate_poly: 证明者发送的单变量多项式
        var: 当前轮次的变量
        round_num: 轮次编号
        """

        # 检查 g_i(0) + g_i(1) = 上一轮的期望值
        val_0 = univariate_poly.subs(var, 0)
        val_1 = univariate_poly.subs(var, 1)

        if round_num == 0:
            expected_sum = self.claimed_sum
        else:
            expected_sum = self.last_polynomial_value

        # 使用容差比较而不是严格相等
        tolerance = 1e-10  # 设置容差为 10^-10
        actual_sum = val_0 + val_1
        
        if abs(float(actual_sum) - float(expected_sum)) > tolerance:
            return False, f"Round {round_num}: g({0}) + g({1}) = {actual_sum} ≠ {expected_sum}"
        
        #产生随机数 
        challenge = self.generate_challenge()
        self.challenge_values.append(challenge)
    
        self.last_polynomial_value = univariate_poly.subs(var, challenge)
    
        return True, f"Round {round_num} passed, challenge = {challenge}"




def sumcheck_protocol(vec, verbose=True):
    """
    完整的sumcheck协议执行
    vec: 输入向量
    verbose: 是否打印详细信息
    """

    # 1. 构建MLE多项式
    poly = mle_polynomial_symbolic(vec)

    # 提取多项式中的所有自由变量（符号）
    variables = list(poly.free_symbols)

    #按变量名字符串排序，确保变量顺序一致
    variables.sort(key=lambda x: x.name)

    
    if verbose:
        print(f"MLE多项式: {poly}")
        print(f"变量: {[str(v) for v in variables]}")


    # 2. 初始化证明者和验证者
    prover = SumcheckProver(poly, variables)
    claimed_sum = prover.compute_claimed_sum()
    
    if verbose:
        print(f"\n声称的总和: {claimed_sum}")
    
    verifier = SumcheckVerifier(len(variables), claimed_sum)
    
    # 3. 执行sumcheck轮次
    if verbose:
        print("\n=== Sumcheck协议执行 ===")
    
    for round_num in range(len(variables)):
        if verbose:
            print(f"\n--- 第 {round_num + 1} 轮 ---")
        
        # 证明者生成单变量多项式
        univariate_poly = prover.generate_univariate_polynomial(
            round_num, verifier.challenge_values
        )
        
        if verbose:
            print(f"证明者发送多项式: {univariate_poly}")
        
        # 验证者验证
        is_valid, message = verifier.verify_round(
            univariate_poly, variables[round_num], round_num
        )
        
        if verbose:
            print(f"验证结果: {message}")
        
        if not is_valid:
            return False, f"验证失败: {message}"
    
    # 4. 最终验证
    # 计算多项式在所有挑战点的值
    final_assignment = {var: val for var, val in zip(variables, verifier.challenge_values)}
    final_value = poly.subs(final_assignment)
    
    if verbose:
        print(f"\n=== 最终验证 ===")
        print(f"挑战点: {verifier.challenge_values}")
        print(f"多项式在挑战点的值: {final_value}")
        print(f"最后一轮期望值: {verifier.last_polynomial_value}")
    
    # 使用容差比较
    tolerance = 1e-10
    if abs(float(final_value) - float(verifier.last_polynomial_value)) <= tolerance:
        if verbose:
            print("✓ Sumcheck协议验证成功!")
        return True, "验证成功"
    else:
        if verbose:
            print("✗ 最终验证失败!")
        return False, "最终验证失败"

    
