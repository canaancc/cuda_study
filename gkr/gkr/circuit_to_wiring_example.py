import numpy as np
from typing import List, Dict, Tuple
from arithmetic_circuit import ArithmeticCircuit
from gkr_data import GKRWiring, GKRProofData

def convert_arithmetic_circuit_to_gkr_wiring(circuit: ArithmeticCircuit) -> GKRWiring:
    """
    将算术电路转换为 GKR Wiring 形式
    
    Args:
        circuit: 算术电路对象
        
    Returns:
        GKRWiring: 对应的 GKR 连线结构
    """
    
    # 1. 分析电路结构，确定每层的变量数
    num_variables_per_layer = analyze_circuit_layers(circuit)
    
    # 2. 创建 GKRWiring 对象
    # 创建对象就是每一个layer 对应的变量数量
    wiring = GKRWiring(num_variables_per_layer)
    
    # 3. 为每一层添加连线函数
    for layer_idx in range(1, len(circuit.layers) + 1):
        left_wire_func, right_wire_func = create_wiring_functions_for_layer(
            circuit, layer_idx
        )
        
        wiring.add_wiring_function(
            layer_from=layer_idx - 1,
            layer_to=layer_idx,
            left_wire_func=left_wire_func,
            right_wire_func=right_wire_func
        )
    
   
    
    return wiring

def analyze_circuit_layers(circuit: ArithmeticCircuit) -> List[int]:
    """
    分析电路每层的变量数
    """

    # 获得输入的数据量
    input_count = len([k for k in circuit.gate_values.keys() if k.startswith('input')])
    # 用 对应的bits 去代表所有的数据
    input_layer_vars = int(np.ceil(np.log2(input_count))) if input_count > 0 else 1
    
    variables_per_layer = [input_layer_vars] # 建立 input的变量数据
    
    # 计算每一层的门数量，确定变量数
    #print("Gate per Layer")
    #print(circuit.layers)
    for layer in circuit.layers:
        #print(layer)
        gate_count = len(layer)
        layer_vars = int(np.ceil(np.log2(gate_count))) if gate_count > 0 else 1
        variables_per_layer.append(layer_vars)
    #print(variables_per_layer)
    return variables_per_layer

def create_wiring_functions_for_layer(circuit: ArithmeticCircuit, layer_idx: int):
    """
    为特定层创建连线函数
    """
    layer_gates = circuit.layers[layer_idx - 1]  # layer_idx 从 1 开始
    print(f"Layer Gates{layer_idx} is {layer_gates}") 
    # 创建门ID到索引的映射
    gate_id_to_index = {}
    for i, gate in enumerate(layer_gates):
        gate_id_to_index[gate['output']] = i
    
    # 创建输入门ID到索引的映射（用于第一层）
    if layer_idx == 1:
        input_ids = [k for k in circuit.gate_values.keys() if k.startswith('input')]
        input_id_to_index = {input_id: i for i, input_id in enumerate(sorted(input_ids))}
    
    def left_wire_func(gate_index):
        """左输入连线函数"""
        if isinstance(gate_index, tuple):
            gate_idx = gate_index[0]
        else:
            gate_idx = gate_index
            
        if gate_idx < len(layer_gates):
            gate = layer_gates[gate_idx]
            left_input_id = gate['inputs'][0]
            
            if layer_idx == 1:  # 连接到输入层
                return (input_id_to_index.get(left_input_id, 0),)
            else:  # 连接到前一层的门
                # 在前一层中查找对应的门索引
                prev_layer_gates = circuit.layers[layer_idx - 2]
                for i, prev_gate in enumerate(prev_layer_gates):
                    if prev_gate['output'] == left_input_id:
                        return (i,)
                return (0,)  # 默认值
        return (0,)
    
    def right_wire_func(gate_index):
        """右输入连线函数"""
        if isinstance(gate_index, tuple):
            gate_idx = gate_index[0]
        else:
            gate_idx = gate_index
            
        if gate_idx < len(layer_gates):
            gate = layer_gates[gate_idx]
            right_input_id = gate['inputs'][1]
            
            if layer_idx == 1:  # 连接到输入层
                return (input_id_to_index.get(right_input_id, 0),)
            else:  # 连接到前一层的门
                # 在前一层中查找对应的门索引
                prev_layer_gates = circuit.layers[layer_idx - 2]
                for i, prev_gate in enumerate(prev_layer_gates):
                    if prev_gate['output'] == right_input_id:
                        return (i,)
                return (0,)  # 默认值
        return (0,)
    print(f"layer{layer_idx}")
    print(f"gate_id_to_index is {gate_id_to_index}")
    print(f"left_wire_func is {left_wire_func}")
    print(f"right_wire_func is {right_wire_func}")
    return left_wire_func, right_wire_func

def demonstrate_conversion():
    """
    演示完整的转换过程
    """
    print("=== 算术电路到 GKR Wiring 转换示例 ===")
    
    # 1. 创建示例算术电路：计算 (a+b) * (c+d)
    circuit = ArithmeticCircuit(num_layers=2)
    
    # 设置输入
    input_values = np.array([1, 2, 3, 4])  # a=1, b=2, c=3, d=4
    circuit.set_input_layer(input_values)
    
    # 第一层：两个加法门
    layer1_gates = [
        {
            'type': 'add',
            'inputs': ['input_0', 'input_1'],  # a + b
            'output': 'gate1_add'
        },
        {
            'type': 'add', 
            'inputs': ['input_2', 'input_3'],  # c + d
            'output': 'gate2_add'
        }
    ]
    circuit.add_layer(layer1_gates)
    
    # 第二层：一个乘法门
    layer2_gates = [
        {
            'type': 'mul',
            'inputs': ['gate1_add', 'gate2_add'],  # (a+b) * (c+d)
            'output': 'final_result'
        }
    ]
    circuit.add_layer(layer2_gates)
    
    print("\n--- 原始算术电路结构 ---")
    print(f"输入层: {list(circuit.gate_values.keys())}")
    for i, layer in enumerate(circuit.layers):
        print(f"第{i+1}层: {layer}")
    
    # 2. 转换为 GKR Wiring
    wiring = convert_arithmetic_circuit_to_gkr_wiring(circuit)
    
    print("\n--- 转换后的 GKR Wiring 结构 ---")
    print(f"每层变量数: {wiring.num_variables_per_layer}")
    print(f"层间连接: {wiring.layer_connections}")
    print(f"连线函数数量: {len(wiring.wiring_functions)}")
    
    # 3. 验证连线函数
    print("\n--- 连线函数验证 ---")
    for layer in range(1, len(wiring.num_variables_per_layer)):
        layer_size = wiring.get_layer_size(layer)
        print(f"\n第{layer}层 (大小: {layer_size})")
        
        for gate_idx in range(min(layer_size, 4)):  # 只显示前4个门
            left_input, right_input = wiring.evaluate_wiring(layer, (gate_idx,))
            print(f"  门{gate_idx}: 左输入={left_input}, 右输入={right_input}")
    
    # 4. 创建 GKR 证明数据结构
    proof_data = GKRProofData()
    proof_data.circuit_info = {
        "num_layers": circuit.num_layers + 1,  # 包括输入层
        "num_variables_per_layer": wiring.num_variables_per_layer,
        "circuit_depth": circuit.num_layers,
        "total_gates": sum(len(layer) for layer in circuit.layers)
    }
    proof_data.add_wiring_info(wiring)
    
    print("\n--- GKR 证明数据结构 ---")
    print(f"电路信息: {proof_data.circuit_info}")
    print(f"连线信息: {proof_data.wiring_info}")
    
    # 5. 验证连线有效性
    is_valid = wiring.validate_wiring()
    print(f"\n连线验证结果: {'✓ 有效' if is_valid else '✗ 无效'}")
    
    return circuit, wiring, proof_data

def create_more_complex_example():
    """
    创建更复杂的电路示例：计算 (a*b + c*d) + (e+f)
    使用equal门避免跨层连接
    """
    print("\n\n=== 复杂电路示例：(a*b + c*d) + (e+f) ===\n使用equal门避免跨层连接")
    
    circuit = ArithmeticCircuit(num_layers=3)
    
    # 设置输入：6个变量
    input_values = np.array([1, 2, 3, 4, 5, 6])  # a,b,c,d,e,f
    circuit.set_input_layer(input_values)
    
    # 第一层：两个乘法门 + 一个加法门
    layer1_gates = [
        {'type': 'mul', 'inputs': ['input_0', 'input_1'], 'output': 'gate1_mul'},  # a*b
        {'type': 'mul', 'inputs': ['input_2', 'input_3'], 'output': 'gate2_mul'},  # c*d  
        {'type': 'add', 'inputs': ['input_4', 'input_5'], 'output': 'gate3_add'}   # e+f
    ]
    circuit.add_layer(layer1_gates)
    
    # 第二层：一个加法门 + 一个恒等门
    layer2_gates = [
        {'type': 'add', 'inputs': ['gate1_mul', 'gate2_mul'], 'output': 'gate4_add'},  # a*b + c*d
        {'type': 'equal', 'inputs': ['gate3_add', 'gate3_add'], 'output': 'gate5_identity'}  # e+f (恒等门)
    ]
    circuit.add_layer(layer2_gates)
    
    # 第三层：最终加法
    layer3_gates = [
        {'type': 'add', 'inputs': ['gate4_add', 'gate5_identity'], 'output': 'final_result'}  # (a*b + c*d) + (e+f)
    ]
    circuit.add_layer(layer3_gates)
    
    # 转换为 GKR Wiring
    wiring = convert_arithmetic_circuit_to_gkr_wiring(circuit)
    
    print(f"复杂电路的 GKR 结构:")
    print(f"  每层变量数: {wiring.num_variables_per_layer}")
    print(f"  层间连接: {wiring.layer_connections}")
    
    # 验证连线函数
    print("\n--- 连线函数验证 ---")
    for layer in range(1, len(wiring.num_variables_per_layer)):
        layer_size = wiring.get_layer_size(layer)
        print(f"\n第{layer}层 (大小: {layer_size})")
        
        for gate_idx in range(min(layer_size, 4)):  # 只显示前4个门
            left_input, right_input = wiring.evaluate_wiring(layer, (gate_idx,))
            print(f"  门{gate_idx}: 左输入={left_input}, 右输入={right_input}")
    
    # 创建 GKR 证明数据结构
    proof_data = GKRProofData()
    proof_data.circuit_info = {
        "num_layers": circuit.num_layers + 1,  # 包括输入层
        "num_variables_per_layer": wiring.num_variables_per_layer,
        "circuit_depth": circuit.num_layers,
        "total_gates": sum(len(layer) for layer in circuit.layers)
    }
    proof_data.add_wiring_info(wiring)
    
    print("\n--- GKR 证明数据结构 ---")
    print(f"电路信息: {proof_data.circuit_info}")
    print(f"连线信息: {proof_data.wiring_info}")
    
    # 验证连线有效性
    is_valid = wiring.validate_wiring()
    print(f"\n连线验证结果: {'✓ 有效' if is_valid else '✗ 无效'}")
    
    # 计算并验证结果
    result = circuit.evaluate_circuit()
    expected = (1*2 + 3*4) + (5+6)  # (2 + 12) + 11 = 25
    print(f"\n计算结果验证:")
    print(f"  电路计算: {result['final_result']}")
    print(f"  期望结果: {expected}")
    print(f"  结果正确: {result['final_result'] == expected}")
    
    return circuit, wiring, proof_data

if __name__ == "__main__":
    # 运行基本示例
   # circuit1, wiring1, proof_data1 = demonstrate_conversion()
    
    # 运行复杂示例
    circuit2, wiring2 , proof_data2= create_more_complex_example()