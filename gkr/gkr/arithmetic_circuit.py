import numpy as np
from typing import List, Dict
class ArithmeticCircuit:
    """
    算术电路表示类
    支持加法门和乘法门的分层电路
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layers = []  # 每层的门信息
        self.wiring = []  # 连线信息
        self.gate_values = {}  # 门的计算值
        
    def add_layer(self, layer_gates: List[Dict]):
        """
        添加一层门
        layer_gates: [{'type': 'add'/'mul', 'inputs': [gate_id1, gate_id2], 'output': gate_id}]
        """
        self.layers.append(layer_gates)
        
    def set_input_layer(self, input_values: np.ndarray):
        """
        设置输入层的值
        """
        for i, value in enumerate(input_values):
            self.gate_values[f"input_{i}"] = value
            
    def evaluate_circuit(self) -> Dict[str, float]:
        """
        计算整个电路的值
        """
        for layer_idx, layer in enumerate(self.layers):

            for gate in layer:
                if gate['type'] == 'add':
                    val1 = self.gate_values[gate['inputs'][0]]
                    val2 = self.gate_values[gate['inputs'][1]]
                    self.gate_values[gate['output']] = val1 + val2
                elif gate['type'] == 'mul':
                    val1 = self.gate_values[gate['inputs'][0]]
                    val2 = self.gate_values[gate['inputs'][1]]
                    self.gate_values[gate['output']] = val1 * val2
                elif gate['type'] == 'equal':
                    # 恒等门：输出第一个输入值
                    val1 = self.gate_values[gate['inputs'][0]]
                    self.gate_values[gate['output']] = val1
                print(f"gate_value i {self.gate_values}")
                    
        return self.gate_values


def create_example_circuit():
    """
    创建一个计算 (a+b) * (c+d) 的算术电路
    
    电路结构：
    输入层: a, b, c, d
    第1层: gate1 = a+b, gate2 = c+d  
    第2层: gate3 = gate1 * gate2
    """
    circuit = ArithmeticCircuit(num_layers=2)

    # layer1
    layer1_gate = [
        {
            'type' :'add',
            'inputs' : ['input_0', 'input_1'],
            'output' : 'gate1_add'
        },
        {
            'type' : 'add',
            'inputs' : ['input_2', 'input_3'],
            'output' : 'gate2_add'
        }
    ]
    circuit.add_layer(layer1_gate)

    # layer2
    layer2_gate = [
        {
            'type' : 'mul',
            'inputs' : ['gate1_add', 'gate2_add'],
            'output' : 'final_result'
        }
    ]
    circuit.add_layer(layer2_gate)
    return circuit

def run_circuit_example():
    """
    运行算术电路示例
    """
    print("=== 算术电路示例：计算 (a+b) * (c+d) ===")    

    #1. create circuit
    circuit = create_example_circuit()

    # 2. set input 
    input_values = np.array([1,2,3,4])
    circuit.set_input_layer(input_values)
    print(f"输入值: a={input_values[0]}, b={input_values[1]}, c={input_values[2]}, d={input_values[3]}")

    # 3. 计算电路
    print("\n--- 电路计算过程 ---")
    
    # 显示初始状态
    print("初始门值:", {k: v for k, v in circuit.gate_values.items() if k.startswith('input')})
    
    # 执行计算
    result = circuit.evaluate_circuit()

    print(result)
    
    # 4. 显示每层的计算结果
    print("\n--- 分层计算结果 ---")
    print(f"第1层结果:")
    print(f"  gate1_add (a+b): {result['gate1_add']}")
    print(f"  gate2_add (c+d): {result['gate2_add']}")
    
    print(f"\n第2层结果:")
    print(f"  final_result ((a+b)*(c+d)): {result['final_result']}")
    
    # 5. 验证结果
    expected = (input_values[0] + input_values[1]) * (input_values[2] + input_values[3])
    print(f"\n--- 验证 ---")
    print(f"手工计算: ({input_values[0]}+{input_values[1]}) * ({input_values[2]}+{input_values[3]}) = {expected}")
    print(f"电路计算: {result['final_result']}")
    print(f"结果正确: {result['final_result'] == expected}")
    
    return circuit, result


if __name__ == "__main__":
    circuit1, result1 = run_circuit_example()