import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id, dispatch_core_config=ttnn.device.DispatchCoreConfig())

# 32*8是一个core应该处理的量
# 如果大于32*8*8，应该效仿attention.py，将torch_input_tensor_a reshape到batch维度
# TODO: K值有问题，最后再改吧
# torch_input_tensor_a = torch.rand(32*8*2, 64*8*4, dtype=torch.float32)

# 定义张量尺寸
R = 32 * 8 * 2 * 2  # 行数 = 512
C = 64 * 8 * 4 * 2  # 列数 = 2048
N = 8  # 循环的模数 (0到7)
# 1. 创建从 0 到 C-1 的序列 (形状: [2048])
# 例如：[0, 1, 2, ..., 2047]
sequence = torch.arange(C)
# 2. 对序列进行取模操作 (形状: [2048])
# 结果：[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, ...]
row_pattern = sequence % N + 1
# torch_input_tensor_a = row_pattern.unsqueeze(0).expand(R, C).to(torch.float32)
torch_input_tensor_a = torch.rand(R, C, dtype=torch.float32)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


rmsnorm = ttnn.rms_norm(input_tensor_a, epsilon=1e-5)
# torch_output_tensor = ttnn.to_torch(output_tensor)

# torch_input_tensor_b = torch.rand(64*8*4, 32*8*2, dtype=torch.float32)
torch_input_tensor_b = torch.ones(C, R, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

output_tensor = ttnn.exp(input_tensor_b)


program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=8,
    per_core_N=8,
    out_block_h=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

program_config_fusenorm = ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=8,
    per_core_N=8,
    out_block_h=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

print("program_config_fusenorm=ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm")

liner_output_tensor = ttnn.linear(rmsnorm, input_tensor_b, program_config=program_config)
torch_linear_output_tensor = ttnn.to_torch(liner_output_tensor)
torch.set_printoptions(threshold=10000)
print(torch_linear_output_tensor[0])


output_tensor = ttnn.exp(input_tensor_b)
linear_norm_output_tensor = ttnn.linear_norm(
    input_tensor_a, input_tensor_b, gamma=None, epsilon=1e-5, program_config=program_config_fusenorm
)
torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)

torch.set_printoptions(threshold=10000)
print(torch_linear_norm_output_tensor[0])


ttnn.close_device(device)


def calculate_elementwise_pcc(tensor1, tensor2):
    """
    计算两个 PyTorch Tensor 元素级别的皮尔逊相关系数 (PCC)。
    它会先将两个 Tensor 展平为一维，然后计算它们之间的相关性。

    Args:
        tensor1 (torch.Tensor): 第一个二维或多维 Tensor。
        tensor2 (torch.Tensor): 第二个与 tensor1 形状相同的 Tensor。

    Returns:
        float: 两个 Tensor 展平后的一维数据之间的 PCC。
    """
    # 1. 确保两个 Tensor 形状相同
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个 Tensor 的形状必须相同才能进行元素级别的相关性计算。")

    # 2. 将 Tensor 展平为一维
    # view(-1) 会把 Tensor 展平为包含所有元素的一维 Tensor
    flat_t1 = tensor1.view(-1)
    flat_t2 = tensor2.view(-1)

    # 3. 计算相关系数
    # torch.corrcoef(input) 接受一个输入矩阵，其中每一行/列是一个变量。
    # 这里我们将 flat_t1 和 flat_t2 堆叠起来，形成一个 2xN 的矩阵，
    # 其中 N 是元素的总数。
    stacked_tensors = torch.stack((flat_t1, flat_t2), dim=0)

    # corr_matrix 是一个 2x2 的相关系数矩阵。
    # corr_matrix[0, 1] 或 corr_matrix[1, 0] 就是我们需要的 PCC。
    corr_matrix = torch.corrcoef(stacked_tensors)

    # 4. 提取 PCC
    pcc = corr_matrix[0, 1].item()

    return pcc


# --- 示例用法 ---
# 假设我们有两个 3x4 的二维 Tensor

# 计算并打印结果
pcc_ab = calculate_elementwise_pcc(torch_linear_output_tensor, torch_linear_norm_output_tensor)

print(f"PCC (A, B) = {pcc_ab:.4f} (预期接近 1.0)")
