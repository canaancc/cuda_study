import sympy as sp
import json
from .Base import SumcheckBase
from .ProofData import ProofData



class NonInteractiveSumcheckVerifier(SumcheckBase):
    """
    非交互式 Sumcheck 协议的验证者
    """
    def __init__(self, start_random, tolerance=1e-10):
        super().__init__(start_random)
        self.tolerance = tolerance
        self.last_polynomial_value = None
        self.expected_challenge = []
    

    def verify_proof(self, proof_data):
        """
        验证完整的 proof
        
        Args:
            proof_data: ProofData 对象或字典
        
        Returns:
            (bool, str): (验证结果, 详细信息)
        """

        #1. 解析proof data
        if isinstance(proof_data, dict):
            proof = ProofData.from_dict(proof_data)
        elif isinstance(proof_data, str):
            proof = ProofData.from_json(proof_data)
        else:
            proof = proof_data 

        #2. 验证 proof 格式
        if not self._validate_proof_format(proof):
            return False, "Proof 格式验证失败"

        # 3. 初始化验证状态
        # 从 start random 和 claimed sum 入手
        self.start_random = proof.protocol_info["start_random"]
        self.last_polynomial_value = proof.protocol_info["claimed_sum"]

        # 4. 验证转录记录
        # 转录记录是确保prover 生成 challenge 的过程是正确的



        # 4. 逐轮验证
        for round_data in proof.rounds_data:
            success, msg = self._verify_round(round_data)
            if not success :
                return False, f"第 {round_data['round_number']} 轮验证失败: {msg}"
        
        # 5. 最终验证
        success, msg = self._final_verification(proof)
        if not success:
            return False, f"最终验证失败: {msg}"
        
        return True, "验证成功"


    def _validate_proof_format(self, proof):
        """
        验证 proof 数据格式的完整性
    
        Returns:
            bool: True 表示格式正确，False 表示格式错误
        """
        try:
            # 检查必要字段
            required_fields = ["protocol_info", "rounds_data", "final_data"]
            for field in required_fields:
                if not hasattr(proof, field) or getattr(proof, field) is None:
                    return False

            # 检查协议信息
            protocol_info = proof.protocol_info
            if not all(key in protocol_info for key in ["claimed_sum", "num_variables", "start_random"]):
                return False

            # 检查轮数一致性
            expected_rounds = protocol_info["num_variables"]
            if len(proof.rounds_data) != expected_rounds:
                return False

            return True
        except Exception:
            return False


    def _verify_round(self, round_data):
        """
        验证单轮数据
        
        Args:
            round_data: 单轮的证明数据
        
        Returns:
            (bool, str): (验证结果, 详细信息)
        """
        try:
            round_num = round_data["round_number"]
            poly_data = round_data["univariate_polynomial"]
            challenge_value = round_data["challenge_value"]
            
            # 1. 重构多项式
            # 将 str 准换为 polynomial
            polynomial_expr = poly_data["expression"]
            polynomial = sp.sympify(polynomial_expr)
            
            # 2. 验证挑战值生成的正确性
            expected_challenge = self.generate_challenge(round_num, polynomial)
            if abs(challenge_value - expected_challenge) > self.tolerance:
                return False, f"挑战值不匹配: 期望 {expected_challenge}, 实际 {challenge_value}"
            
            self.expected_challenge.append(expected_challenge)

            # 3. 验证多项式求值
            g_0 = float(poly_data["evaluations"]["at_0"])
            g_1 = float(poly_data["evaluations"]["at_1"])
            
            # 验证 g(0) + g(1) = 上一轮的期望值
            sum_check = g_0 + g_1
            if abs(sum_check - self.last_polynomial_value) > self.tolerance:
                return False, f"和检查失败: g(0)+g(1)={sum_check}, 期望={self.last_polynomial_value}"
            
            # 4. 计算多项式在挑战点的值
            var = list(polynomial.free_symbols)[0] if polynomial.free_symbols else sp.Symbol('x')

            polynomial_at_challenge = float(polynomial.subs(var, challenge_value))
            
            # 5. 更新状态
            self.last_polynomial_value = polynomial_at_challenge
            
  
            return True, "轮次验证成功"
            
        except Exception as e:
            return False, f"轮次验证异常: {str(e)}"

    def _final_verification(self, proof):
        """
        最终验证步骤
        
        Args:
            proof: 完整的 proof 数据
        
        Returns:
            (bool, str): (验证结果, 详细信息)
        """
        try:
            final_data = proof.final_data
            final_point = final_data["final_point"]
            final_evaluation = final_data["final_evaluation"]
            
            # 检查proof data 的
            expected_challenges = self.expected_challenge
            
            if len(final_point) != len(expected_challenges):
                return False, "最终点维度不匹配"
            
            for i, (actual, expected) in enumerate(zip(final_point, expected_challenges)):
                if abs(actual - expected) > self.tolerance:
                    return False, f"最终点第 {i} 维不匹配: {actual} vs {expected}"
            
            # 验证最终求值是否与最后一轮的多项式值一致
            if abs(final_evaluation - self.last_polynomial_value) > self.tolerance:
                return False, f"最终求值不匹配: {final_evaluation} vs {self.last_polynomial_value}"
            
            return True, "最终验证成功"
            
        except Exception as e:
            return False, f"最终验证异常: {str(e)}"    
