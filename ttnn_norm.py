import torch
import ttnn
import numpy as np


def rmsnorm_no_weight(X, eps=1e-6):
    """
    无 weight 参数的 RMSNorm 实现。
    输入 X 是一个二维矩阵 (R, C)。
    """
    # 1. 计算均方根 (RMS)
    # X.pow(2) 计算元素平方
    # .mean(dim=-1, keepdim=True) 对最后一个维度（列/特征维度 C）求均值，并保持维度
    # .sqrt() 求平方根
    rms = X.pow(2).mean(dim=-1, keepdim=True).sqrt()

    # 2. 归一化：X / (rms + eps)
    Y = X / (rms + eps)

    return Y


device_id = 0
device = ttnn.open_device(device_id=device_id, dispatch_core_config=ttnn.device.DispatchCoreConfig())

# 32*8是一个core应该处理的量
# 如果大于32*8*8，应该效仿attention.py，将torch_input_tensor_a reshape到batch维度
# TODO: K值有问题，最后再改吧
# torch_input_tensor_a = torch.rand(32*8*2, 64*8*4, dtype=torch.float32)

# 定义张量尺寸
R = 32 * 8 * 2 * 2  # 行数 = 512
C = 64 * 8 * 4 * 2  # 列数 = 4096
N = 4096  # 循环的模数 (0到7)
# 1. 创建从 0 到 C-1 的序列 (形状: [2048])
# 例如：[0, 1, 2, ..., 2047]
sequence = torch.arange(C)
# 2. 对序列进行取模操作 (形状: [2048])
# 结果：[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, ...]
row_pattern = (sequence % N + 1) * 0.001
# torch_input_tensor_a = row_pattern.unsqueeze(0).expand(R, C).to(torch.float32)
# torch_input_tensor_a = torch.rand(R, C, dtype=torch.float16)

# 4. 创建一个形状为 [R, C] 的全零张量，用于存放结果
torch_input_tensor_a = torch.zeros(R, C, dtype=torch.float16)

# 5. 设置第一行 (第 0 行) 的值为 row_pattern_base
torch_input_tensor_a[0, :] = row_pattern

# 6. 循环计算后续每一行：当前行 = 上一行 * 2
# 注意：PyTorch 的乘法是元素级的
for i in range(1, 32):
    # 计算 i 行为 i-1 行的 2 倍
    # 这一步是您要求实现的“每一行都比上一行*2”的逻辑
    torch_input_tensor_a[i, :] = torch_input_tensor_a[i - 1, :] * 3.0

# torch_input_tensor_a = torch.rand(R, C, dtype=torch.float16)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
rmsnorm_torch = rmsnorm_no_weight(torch_input_tensor_a)


rmsnorm = ttnn.rms_norm(input_tensor_a, epsilon=1e-5)
torch_rmsnorm_output_tensor = ttnn.to_torch(rmsnorm)
torch.set_printoptions(threshold=10000)
# print(rmsnorm_torch[0])
print(torch_rmsnorm_output_tensor[0])
# torch_output_tensor = ttnn.to_torch(output_tensor)

# torch_input_tensor_b = torch.rand(64*8*4, 32*8*2, dtype=torch.float32)
torch_input_tensor_b = torch.ones(C, R, dtype=torch.float16)
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

# liner_output_tensor = ttnn.linear(rmsnorm, input_tensor_b, program_config=program_config)
# torch_linear_output_tensor = ttnn.to_torch(liner_output_tensor)
# # torch.set_printoptions(threshold=10000)
# print(torch_linear_output_tensor[0])


# torch_matmul_output_tensor = torch.matmul(rmsnorm_torch, torch_input_tensor_b)

output_tensor = ttnn.exp(input_tensor_b)
compute_kernel_config = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
linear_norm_output_tensor = ttnn.linear_norm(
    input_tensor_a,
    input_tensor_b,
    gamma=None,
    epsilon=1e-5,
    program_config=program_config_fusenorm,
    compute_kernel_config=compute_kernel_config,
)
torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)

# torch.set_printoptions(threshold=10000)
# print(torch_matmul_output_tensor[0])
print(torch_linear_norm_output_tensor[0])


ttnn.close_device(device)


def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


# --- 示例用法 ---
# 假设我们有两个 3x4 的二维 Tensor

# 计算并打印结果
passing, pcc_message = comp_pcc(torch_matmul_output_tensor, torch_linear_norm_output_tensor)

print(f"PCC: {pcc_message}")
