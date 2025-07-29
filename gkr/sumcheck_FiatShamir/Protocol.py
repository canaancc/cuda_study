from .ProofData import ProofData
from .Prover import NonInteractiveSumcheckProver
from .Verifier import NonInteractiveSumcheckVerifier
from mle import mle_polynomial_symbolic
import random
import time
import json

class SumcheckProtocol:
    """
    非交互式 Sumcheck 协议的统一管理类
    提供完整的证明生成和验证流程
    """
    
    def __init__(self, start_random=None, tolerance=1e-10):
        """
        初始化协议
        
        Args:
            start_random: 初始随机种子，如果为 None 则自动生成
            tolerance: 验证时的数值容差
        """
        self.start_random = start_random if start_random is not None else random.randint(1, 2**64)
        self.tolerance = tolerance
        self.prover = None
        self.verifier = None
        
    def setup_for_vec(self, vec):
        """
        为给定向量设置协议
        
        Args:
            vec: 输入向量 (numpy array 或 list)
            
        Returns:
            dict: 协议设置信息
        """
        # 构建 MLE 多项式
        mle_poly = mle_polynomial_symbolic(vec)
        
        # 提取变量并排序
        variables = list(mle_poly.free_symbols)
        variables.sort(key=lambda x: x.name)
        
        # 初始化 Prover 和 Verifier
        self.prover = NonInteractiveSumcheckProver(mle_poly, variables, self.start_random)
        self.verifier = NonInteractiveSumcheckVerifier(self.start_random, self.tolerance)
        
        setup_info = {
            "polynomial": str(mle_poly),
            "variables": [str(v) for v in variables],
            "num_variables": len(variables),
            "start_random": self.start_random,
            "vector_sum": sum(vec),
            "claimed_sum": float(self.prover.compute_claimed_sum())
        }
        
        return setup_info
    
     
    def generate_proof(self):
        """
        生成完整的 Sumcheck 证明
        
        Returns:
            ProofData: 证明数据对象
        """
        if self.prover is None:
            raise ValueError("协议未初始化，请先调用 setup_for_vector 或 setup_for_polynomial")
        
        start_time = time.time()
        proof = self.prover.generate_proof()
        end_time = time.time()
        
        # 添加性能统计信息
        performance_stats = {
            "proof_generation_time": end_time - start_time,
            "num_rounds": len(proof.rounds_data),
            "proof_size_bytes": len(proof.to_json().encode('utf-8'))
        }

        
        return proof, performance_stats
    
    def verify_proof(self, proof_data):
        """
        验证 Sumcheck 证明
        
        Args:
            proof_data: ProofData 对象、字典或 JSON 字符串
            
        Returns:
            dict: 验证结果详情
        """
        if self.verifier is None:
            self.verifier = NonInteractiveSumcheckVerifier(
                proof_data.protocol_info["start_random"], 
                self.tolerance
            )
        
        start_time = time.time()
        success, message = self.verifier.verify_proof(proof_data)
        end_time = time.time()
        
        verification_result = {
            "success": success,
            "message": message,
            "verification_time": end_time - start_time,
            "verifier_tolerance": self.tolerance
        }
        
        return verification_result
    
    def run_complete_protocol(self, vec=None, verbose=True):
        """
        运行完整的协议流程：设置 -> 生成证明 -> 验证证明
        
        Args:
            vec: 输入向量（与 polynomial/variables 二选一）
            polynomial: 多项式表达式（与 vec 二选一）
            variables: 变量列表（与 polynomial 配套使用）
            verbose: 是否打印详细信息
            
        Returns:
            dict: 完整的协议执行结果
        """
        try:
            # 1. 协议设置
            if vec is not None:
                setup_info = self.setup_for_vec(vec)
                if verbose:
                    print(f"=== 协议设置（向量模式）===")
                    print(f"向量: {vec}")
            else:
                raise ValueError("必须提供 vec")
            
            if verbose:
                print(f"变量数量: {setup_info['num_variables']}")
                print(f"声称总和: {setup_info['claimed_sum']}")
                print(f"随机种子: {setup_info['start_random']}")
            
            # 2. 生成证明
            if verbose:
                print(f"\n=== 证明生成 ===")
            
            proof, performance_stats = self.generate_proof()
            
            if verbose:
                print(f"证明生成时间: {performance_stats['proof_generation_time']:.4f}s")
                print(f"证明大小: {performance_stats['proof_size_bytes']} bytes")
            
            # 3. 验证证明
            if verbose:
                print(f"\n=== 证明验证 ===")
            
            verification_result = self.verify_proof(proof)
            
            if verbose:
                print(f"验证结果: {'✓ 成功' if verification_result['success'] else '✗ 失败'}")
                print(f"验证信息: {verification_result['message']}")
                print(f"验证时间: {verification_result['verification_time']:.4f}s")
            
            # 4. 返回完整结果
            return {
                "setup_info": setup_info,
                "proof": proof,
                "verification_result": verification_result,
                "protocol_success": verification_result['success']
            }
            
        except Exception as e:
            error_result = {
                "setup_info": None,
                "proof": None,
                "verification_result": {"success": False, "message": f"协议执行异常: {str(e)}"},
                "protocol_success": False
            }
            
            if verbose:
                print(f"✗ 协议执行失败: {str(e)}")
            
            return error_result
    
    def export_proof(self, proof, filepath):
        """
        导出证明到文件
        
        Args:
            proof: ProofData 对象
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(proof.to_json())
    
    def import_proof(self, filepath):
        """
        从文件导入证明
        
        Args:
            filepath: 文件路径
            
        Returns:
            ProofData: 证明数据对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return ProofData.from_json(json_str)
    
    def get_protocol_stats(self):
        """
        获取协议统计信息
        
        Returns:
            dict: 协议统计信息
        """
        stats = {
            "start_random": self.start_random,
            "tolerance": self.tolerance,
            "prover_initialized": self.prover is not None,
            "verifier_initialized": self.verifier is not None
        }
        
        if self.prover:
            stats.update({
                "num_variables": self.prover.num_len,
                "polynomial": str(self.prover.polynomial),
                "variables": [str(v) for v in self.prover.variables]
            })
        
        return stats


# 便捷函数
def sumcheck_for_vector(vec, start_random=None, verbose=True):
    """
    为向量运行完整的 Sumcheck 协议的便捷函数
    
    Args:
        vec: 输入向量
        start_random: 初始随机种子
        verbose: 是否打印详细信息
        
    Returns:
        dict: 协议执行结果
    """
    protocol = SumcheckProtocol(start_random)
    return protocol.run_complete_protocol(vec=vec, verbose=verbose)



def verify_external_proof(proof_data, start_random=None, tolerance=1e-10):
    """
    验证外部证明的便捷函数
    
    Args:
        proof_data: 证明数据（ProofData 对象、字典或 JSON 字符串）
        start_random: 初始随机种子（如果为 None，从 proof 中提取）
        tolerance: 验证容差
        
    Returns:
        dict: 验证结果
    """
    protocol = SumcheckProtocol(start_random, tolerance)
    return protocol.verify_proof(proof_data)