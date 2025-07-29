import hashlib
import json
import random
import sympy as sp

class SumcheckBase:
    """Sumcheck协议的基类，包含共同的方法"""
    def __init__(self, start_random):
        self.start_random = start_random
        self.challenge = []
        self.current_hash = 0

    def generate_challenge(self, round_num, polynomial):
        """
        使用链式哈希生成挑战值
        每轮的哈希输入包含：当前轮次数据 + 上一轮的哈希值
        """
        current_data = {
            "round_num": round_num,
            "polynomial": str(polynomial) if polynomial else "",
            "random" : self.start_random
        } 

        # 获取上一轮的哈希值
        if round_num == 0:
            # 第一轮：使用初始随机种子的哈希
            previous_hash = hashlib.sha256(
                str(self.start_random).encode('utf-8')
            ).hexdigest()
        else:
            # 后续轮次：使用上一轮的累积哈希
            previous_hash = self.current_hash
        
        # 组合当前数据和前一轮哈希
        combine_data = {
            **current_data,
            "previous_hash": previous_hash
        }

        # 生成挑战数值
        challenge_hash = hashlib.sha256(
            json.dumps(combine_data, sort_keys=True).encode('utf-8')
        ).hexdigest()

        # 更新累积哈希
        self.current_hash = challenge_hash
   
        # 转换为数值挑战
        challenge_value = int(challenge_hash[:16], 16) / (2**64)
        
        return challenge_value


