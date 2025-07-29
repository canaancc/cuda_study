
import sympy as sp
from sumcheck import SumcheckProver
from mle import mle_polynomial_symbolic
from .ProofData import ProofData
from .Base import SumcheckBase

class NonInteractiveSumcheckProver(SumcheckProver, SumcheckBase):
    def __init__(self, polynomial, variables, start_random) -> None:
        """
        初始化非交互式 Sumcheck 证明者
        
        Args:
            polynomial: sympy多项式表达式
            variables: 变量列表，按顺序排列 [x0, x1, x2, ...]
            start_random: 初始随机种子
        """
        SumcheckProver.__init__(self, polynomial, variables)
        SumcheckBase.__init__(self, start_random)
    
    def generate_proof(self):
        """
        生成完整的非交互式 Sumcheck 证明
        
        Returns:
            ProofData: 包含所有验证信息的证明数据
        """
        # 1. 初始化 ProofData
        proof = ProofData()
        
        # 2. 设置协议基本信息
        claimed_sum = self.compute_claimed_sum()
        proof.protocol_info = {
            "claimed_sum": float(claimed_sum),
            "num_variables": self.num_len,
            "num_rounds": self.num_len,
            "start_random": self.start_random
        }

        # 3. 执行协议轮次
        challenge_values = []
        last_polynomial_value = claimed_sum
                
        for round_num in range(self.num_len):
            # 生成当前轮的单变量多项式
            univariate_poly = self.generate_univariate_polynomial(round_num, challenge_values)
            
            # 基于多项式和历史转录生成挑战值
            challenge_value = self.generate_challenge(round_num, univariate_poly)
            challenge_values.append(challenge_value)
            
            # 添加轮次数据到 proof
            proof.add_round_data(round_num, univariate_poly, challenge_value)
            
            # 计算验证检查数据
            var = self.variables[round_num]
            g_0 = float(univariate_poly.subs(var, 0))
            g_1 = float(univariate_poly.subs(var, 1))
            sum_check = g_0 + g_1
            
            # 更新轮次数据中的验证信息
            proof.rounds_data[-1]["verification_check"] = {
                "sum_check": sum_check,
                "expected_sum": float(last_polynomial_value)
            }
            
            # 计算多项式在挑战点的值，为下一轮做准备
            last_polynomial_value = univariate_poly.subs(var, challenge_value)
        
        # 4. 设置最终验证数据
        proof.final_data = {
            "final_point": challenge_values,
            "final_evaluation": float(last_polynomial_value)
        }

        # 5. 内部检查
        success , msg = self.verify_internal_consistency(proof) 

        if success:
            print("Proof generation Success")
            return proof
        else :
            print("Proof generation Fail!")
            print(msg)
            return 0


    def verify_internal_consistency(self, proof):
        """
        内部一致性检查：验证生成的 proof 是否自洽
        
        Args:
            proof: ProofData 对象
            
        Returns:
            (bool, str): (验证结果, 详细信息)
        """
        try:
            # 检查轮数一致性
            if len(proof.rounds_data) != proof.protocol_info["num_variables"]:
                return False, "轮数与变量数不匹配"
            
            # 检查每轮的和检查
            expected_sum = proof.protocol_info["claimed_sum"]
            
            for i, round_data in enumerate(proof.rounds_data):
                verification = round_data["verification_check"]
                
                if abs(verification["sum_check"] - verification["expected_sum"]) > 1e-10:
                    return False, f"第 {i} 轮和检查失败"
                
                expected_sum = verification["sum_check"]
            
            # 检查最终点维度
            if len(proof.final_data["final_point"]) != proof.protocol_info["num_variables"]:
                return False, "最终点维度不匹配"
            
            return True, "内部一致性检查通过"
            
        except Exception as e:
            return False, f"一致性检查异常: {str(e)}"
        
    def get_proof_summary(self, proof):
        """
        获取证明的摘要信息
        
        Args:
            proof: ProofData 对象
            
        Returns:
            dict: 证明摘要
        """
        return {
            "claimed_sum": proof.protocol_info["claimed_sum"],
            "num_variables": proof.protocol_info["num_variables"],
            "num_rounds": len(proof.rounds_data),
            "final_point": proof.final_data["final_point"],
            "final_evaluation": proof.final_data["final_evaluation"],
            "proof_size_bytes": len(proof.to_json().encode('utf-8'))
        }
    

        