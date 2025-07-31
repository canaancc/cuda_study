import numpy as np
from typing import List, Dict, Tuple, Callable, Any
import sympy as sp
import json
import time

class GKRWiring:
    """
    GKR协议中的连线函数
    定义电路层之间的连接关系
    """
    def __init__(self, num_variables_per_layer: List[int]):
        self.num_variables_per_layer = num_variables_per_layer
        self.wiring_functions = {}  # 存储连线函数
        self.layer_connections = {}  # 存储层间连接关系
        
    def add_wiring_function(self, layer_from: int, layer_to: int, 
                           left_wire_func: Callable, right_wire_func: Callable):
        """
        添加连线函数
        left_wire_func, right_wire_func: 定义左右输入的连线关系
        """
        key = (layer_from, layer_to)
        self.wiring_functions[key] = {
            'left_wire': left_wire_func,
            'right_wire': right_wire_func
        }
        
        # 记录层间连接
        if layer_to not in self.layer_connections:
            self.layer_connections[layer_to] = []
        self.layer_connections[layer_to].append(layer_from)
        
    def evaluate_wiring(self, layer: int, gate_index: Tuple) -> Tuple:
        """
        计算特定门的输入连线
        返回 (left_input_index, right_input_index)
        """
        if layer == 0:  # 输入层没有连线
            return gate_index, gate_index
            
        # 查找前一层的连线函数
        prev_layer = layer - 1
        key = (prev_layer, layer)
        
        if key not in self.wiring_functions:
            # 默认连线：直接连接
            return gate_index, gate_index
            
        wiring = self.wiring_functions[key]
        left_input = wiring['left_wire'](gate_index)
        right_input = wiring['right_wire'](gate_index)
        
        return left_input, right_input
    
    def get_layer_size(self, layer: int) -> int:
        """
        获取指定层的大小（门的数量）
        """
        if layer < len(self.num_variables_per_layer):
            return 2 ** self.num_variables_per_layer[layer]
        return 0
    
    def validate_wiring(self) -> bool:
        """
        验证连线函数的有效性
        """
        for (layer_from, layer_to), wiring in self.wiring_functions.items():
            # 检查层索引有效性
            if layer_from >= len(self.num_variables_per_layer) or layer_to >= len(self.num_variables_per_layer):
                return False
            
            # 检查连线函数是否可调用
            if not callable(wiring['left_wire']) or not callable(wiring['right_wire']):
                return False
                
        return True

class GKRProofData:
    """
    GKR协议的证明数据结构
    """
    def __init__(self):
        self.circuit_info = {
            "num_layers": None,
            "num_variables_per_layer": [],
            "circuit_depth": None,
            "total_gates": None
        }
        self.layer_proofs = []  # 每层的sumcheck证明
        self.wiring_info = None  # 连线信息
        self.final_layer_values = None
        self.protocol_transcript = []
        self.performance_stats = {}
        
    def add_layer_proof(self, layer_idx: int, sumcheck_proof: Dict[str, Any]):
        """
        添加某层的sumcheck证明
        """
        self.layer_proofs.append({
            "layer": layer_idx,
            "proof": sumcheck_proof,
            "timestamp": time.time(),
            "layer_size": len(sumcheck_proof.get('polynomial_evaluations', []))
        })
        
    def add_wiring_info(self, wiring: GKRWiring):
        """
        添加连线信息
        """
        self.wiring_info = {
            "num_variables_per_layer": wiring.num_variables_per_layer,
            "layer_connections": wiring.layer_connections,
            "total_wiring_functions": len(wiring.wiring_functions)
        }
        
    def set_performance_stats(self, stats: Dict[str, float]):
        """
        设置性能统计信息
        """
        self.performance_stats = stats
        
    def to_json(self) -> str:
        """
        序列化为JSON
        """
        data = {
            "circuit_info": self.circuit_info,
            "layer_proofs": self.layer_proofs,
            "wiring_info": self.wiring_info,
            "final_layer_values": self.final_layer_values,
            "protocol_transcript": self.protocol_transcript,
            "performance_stats": self.performance_stats
        }
        return json.dumps(data, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str):
        """
        从JSON反序列化
        """
        data = json.loads(json_str)
        proof = cls()
        proof.circuit_info = data.get("circuit_info", {})
        proof.layer_proofs = data.get("layer_proofs", [])
        proof.wiring_info = data.get("wiring_info", None)
        proof.final_layer_values = data.get("final_layer_values", None)
        proof.protocol_transcript = data.get("protocol_transcript", [])
        proof.performance_stats = data.get("performance_stats", {})
        return proof
    
    def get_proof_size(self) -> int:
        """
        获取证明大小（字节）
        """
        return len(self.to_json().encode('utf-8'))
    
    def validate_proof_structure(self) -> Tuple[bool, str]:
        """
        验证证明结构的完整性
        """
        # 检查基本信息
        if not self.circuit_info.get("num_layers"):
            return False, "缺少电路层数信息"
            
        if not self.circuit_info.get("num_variables_per_layer"):
            return False, "缺少每层变量数信息"
            
        # 检查层证明
        expected_layers = self.circuit_info["num_layers"]
        if len(self.layer_proofs) != expected_layers:
            return False, f"层证明数量不匹配：期望{expected_layers}，实际{len(self.layer_proofs)}"
            
        # 检查连线信息
        if self.wiring_info is None:
            return False, "缺少连线信息"
            
        return True, "证明结构验证通过"

class GKRChallenge:
    """
    GKR协议中的挑战值管理
    """
    def __init__(self, start_random: int = None):
        self.start_random = start_random or np.random.randint(1, 2**32)
        self.layer_challenges = {}  # 每层的挑战值
        self.sumcheck_challenges = {}  # sumcheck的挑战值
        self.transcript_hash = None
        
    def generate_layer_challenge(self, layer_idx: int, layer_proof: Dict) -> List[float]:
        """
        为特定层生成挑战值
        """
        import hashlib
        
        # 构建挑战数据
        challenge_data = {
            "layer": layer_idx,
            "proof_hash": hash(str(layer_proof)),
            "start_random": self.start_random,
            "previous_challenges": list(self.layer_challenges.keys())
        }
        
        # 生成哈希
        challenge_str = json.dumps(challenge_data, sort_keys=True)
        challenge_hash = hashlib.sha256(challenge_str.encode()).hexdigest()
        
        # 转换为数值挑战
        num_variables = layer_proof.get('num_variables', 1)
        challenges = []
        
        for i in range(num_variables):
            # 使用哈希的不同部分生成多个挑战值
            hash_part = challenge_hash[i*8:(i+1)*8]
            challenge_val = int(hash_part, 16) / (2**32)
            challenges.append(challenge_val)
            
        self.layer_challenges[layer_idx] = challenges
        return challenges
    
    def get_challenge_transcript(self) -> Dict:
        """
        获取完整的挑战转录
        """
        return {
            "start_random": self.start_random,
            "layer_challenges": self.layer_challenges,
            "sumcheck_challenges": self.sumcheck_challenges
        }
    
    def verify_challenge_consistency(self, expected_transcript: Dict) -> bool:
        """
        验证挑战值的一致性
        """
        current_transcript = self.get_challenge_transcript()
        return current_transcript == expected_transcript

# 便捷函数
def create_identity_wiring(num_layers: int, variables_per_layer: List[int]) -> GKRWiring:
    """
    创建恒等连线（每个门直接连接到前一层对应位置的门）
    """
    wiring = GKRWiring(variables_per_layer)
    
    for layer in range(1, num_layers):
        def left_wire(gate_idx):
            return gate_idx
        
        def right_wire(gate_idx):
            return gate_idx
            
        wiring.add_wiring_function(layer-1, layer, left_wire, right_wire)
    
    return wiring

def create_binary_tree_wiring(num_layers: int, variables_per_layer: List[int]) -> GKRWiring:
    """
    创建二叉树连线（每个门连接到前一层的两个门）
    """
    wiring = GKRWiring(variables_per_layer)
    
    for layer in range(1, num_layers):
        def left_wire(gate_idx):
            if isinstance(gate_idx, tuple):
                gate_idx = gate_idx[0]
            return (gate_idx * 2,)
        
        def right_wire(gate_idx):
            if isinstance(gate_idx, tuple):
                gate_idx = gate_idx[0]
            return (gate_idx * 2 + 1,)
            
        wiring.add_wiring_function(layer-1, layer, left_wire, right_wire)
    
    return wiring