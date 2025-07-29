

class ProofData:
    """
    非交互式 Sumcheck 协议的 Proof 数据结构
    包含 Verifier 验证所需的所有信息
    """
    def __init__(self):
        # 1. 协议基本信息
        self.protocol_info = {
            "claimed_sum": None,           # 声称的总和值
            "num_variables": None,         # 变量数量
            "num_rounds": None,            # 协议轮数
            "start_random": None           # 初始随机种子
        }
        
        # 2. 每轮的证明数据
        self.rounds_data = []              # 列表，每个元素是一轮的数据
        
        # 3. 最终验证数据
        self.final_data = {
            "final_point": None,           # 最终挑战点 (r1, r2, ..., rn)
            "final_evaluation": None       # 多项式在最终点的值 f(r1, r2, ..., rn)
        }
        

    def add_round_data(self, round_num, univariate_poly, challenge_value):
        """
        添加一轮的证明数据
        """
        round_data = {
            "round_number": round_num,
            "univariate_polynomial": {
                "expression": str(univariate_poly),    # 多项式表达式字符串
                "coefficients": self._extract_coefficients(univariate_poly),  # 系数列表
                "degree": sp.degree(univariate_poly),   # 多项式度数
                "evaluations": {
                    "at_0": float(univariate_poly.subs(list(univariate_poly.free_symbols)[0], 0)),
                    "at_1": float(univariate_poly.subs(list(univariate_poly.free_symbols)[0], 1))
                }
            },

            "challenge_value": challenge_value,     # 该轮生成的挑战值
            "verification_check": {
                "sum_check": None,                   # g(0) + g(1) 的值
                "expected_sum": None                 # 期望的和值
            }
        }
        self.rounds_data.append(round_data)

    
    def _extract_coefficients(self, polynomial):
        """
        提取多项式的系数
        """
        if not polynomial.free_symbols:
            return [float(polynomial)]
        
        var = list(polynomial.free_symbols)[0]
        poly_expanded = sp.expand(polynomial)
        coeffs = []
        
        # 提取各次项系数
        degree = sp.degree(poly_expanded, var)
        for i in range(degree + 1):
            coeff = poly_expanded.coeff(var, i)
            coeffs.append(float(coeff) if coeff else 0.0)
        
        return coeffs
    
    def to_dict(self):
        """
        将 Proof 数据转换为字典格式，便于序列化
        """
        return {
            "protocol_info": self.protocol_info,
            "rounds_data": self.rounds_data,
            "final_data": self.final_data,
            "transcript": self.transcript
        }
    
    def to_json(self):
        """
        将 Proof 数据序列化为 JSON 字符串
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)
    


    @classmethod
    def from_dict(cls, data_dict):
        """
        从字典创建 ProofData 对象
        """
        proof = cls()
        proof.protocol_info = data_dict["protocol_info"]
        proof.rounds_data = data_dict["rounds_data"]
        proof.final_data = data_dict["final_data"]
        proof.transcript = data_dict["transcript"]
        return proof
    
    @classmethod
    def from_json(cls, json_str):
        """
        从 JSON 字符串创建 ProofData 对象
        """
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)
    
 