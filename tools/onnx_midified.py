import onnx
from onnx import helper, TensorProto


def dims_to_shape(dim_list):
    """将 TensorShapeProto.Dim 列表转换为 int/str 形状列表"""
    shape = []
    for dim in dim_list:
        if dim.dim_value != 0:  # 具体数值维度（如 32, 64）
            shape.append(dim.dim_value)
        else:  # 符号化维度（如 "batch_size"）
            shape.append(dim.dim_param)
    return shape


def convert_node_inputs_to_float16(onnx_path, output_path, target_node_names):
    """将目标节点（如 Div、Mod）的所有 int64 输入转为 float16，确保张量名称唯一"""
    # 1. 加载模型并收集张量信息
    model = onnx.load(onnx_path)
    graph = model.graph
    nodes = list(graph.node)
    tensor_info = {}  # 存储张量类型和形状信息

    # 收集所有张量信息（输入、输出、中间张量、初始化张量）
    all_tensors = list(graph.input) + list(graph.output) + list(graph.value_info)
    for t in all_tensors:
        tensor_info[t.name] = t
    for init in graph.initializer:
        elem_type = init.data_type
        tensor_info[init.name] = helper.make_tensor_value_info(
            init.name, elem_type, init.dims
        )

    # 生成唯一名称的计数器（确保 SSA 合规）
    cast_counter = 0

    # 2. 遍历每个目标节点，处理其输入
    for node_name in target_node_names:
        # 定位目标节点（Div 或 Mod）
        target_node = next((n for n in nodes if n.name == node_name), None)
        if not target_node:
            raise ValueError(f"未找到节点 {node_name}")

        updated_inputs = []
        for inp_name in target_node.input:
            # 跳过无类型信息的张量
            if inp_name not in tensor_info:
                print(f"警告：张量 {inp_name} 无类型信息，跳过转换")
                updated_inputs.append(inp_name)
                continue

            # 检查输入是否为 int64 类型
            tensor_type = tensor_info[inp_name].type.tensor_type.elem_type
            if tensor_type != TensorProto.INT64:
                updated_inputs.append(inp_name)
                continue

            # 3. 生成全局唯一的 Cast 输出名称（关键修改：避免重复）
            global_unique_id = f"cast_{cast_counter}"  # 递增计数器确保唯一
            cast_output_name = f"{inp_name}_to_{node_name}_float16_{global_unique_id}"
            cast_counter += 1  # 计数器递增

            # 创建 Cast 节点（int64 → float16）
            cast_node = helper.make_node(
                op_type="Cast",
                inputs=[inp_name],
                outputs=[cast_output_name],
                name=f"Cast_{global_unique_id}",  # 节点名称也唯一
                to=TensorProto.FLOAT16
            )

            # 4. 记录 Cast 输出的张量信息
            dims = tensor_info[inp_name].type.tensor_type.shape.dim
            cast_shape = dims_to_shape(dims)
            cast_tensor_info = helper.make_tensor_value_info(
                cast_output_name,
                TensorProto.FLOAT16,
                cast_shape
            )
            graph.value_info.append(cast_tensor_info)
            tensor_info[cast_output_name] = cast_tensor_info

            # 5. 插入 Cast 节点到目标节点之前
            target_idx = nodes.index(target_node)
            nodes.insert(target_idx, cast_node)

            # 6. 更新目标节点的输入
            updated_inputs.append(cast_output_name)

        # 替换目标节点的输入列表
        target_node.input[:] = updated_inputs

    # 7. 更新图节点并验证（确保 SSA 合规）
    graph.node.clear()
    graph.node.extend(nodes)

    # 8. 验证模型（此时应通过 SSA 检查）
    try:
        onnx.checker.check_model(model)
        print("模型验证通过，符合 SSA 规则")
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print(f"模型验证失败：{e}")
        return

    # 9. 保存模型
    onnx.save(model, output_path)
    print(f"修改完成，模型已保存至：{output_path}")


# 使用示例
if __name__ == "__main__":
    input_onnx = r"D:\python_work\ultralytics\weights\yolov10s.onnx"  # 原始模型路径
    output_onnx = r"D:\python_work\ultralytics\weights\yolov10s_modify.onnx"  # 输出路径
    # 需根据实际模型中的 Mod 节点名称修改 mod_node_name（例如 Mod_343）
    target_nodes = ["Div_341", "Mod_362"]
    convert_node_inputs_to_float16(
        input_onnx,
        output_onnx,
        target_nodes
    )
