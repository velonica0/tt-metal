import torch
import ttnn
import numpy as np
import math


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


def rmsnorm_with_weight(X, weight, eps=1e-6):
    """
    带 weight 参数的 RMSNorm 实现。
    输入 X: (R, C)
    输入 weight: (C,) - 其长度应与 X 的特征维度相同
    """
    # 1. 计算均方根 (RMS)
    # 这里的计算逻辑保持不变
    rms = X.pow(2).mean(dim=-1, keepdim=True).sqrt()

    # 2. 归一化并应用 weight
    # 公式: Y = (X / (rms + eps)) * weight
    Y = (X / (rms + eps)) * weight

    return Y


def create_dram_sharded_mem_config(k, n):
    dram_weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(12 - 1, 0),
            )
        }
    )
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 12  # WH has 12 dram cores, P150 has 8, P100 has 7
    padded_size = math.ceil(n / (32 * dram_cores)) * (32 * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


device_id = 1
device = ttnn.open_device(device_id=device_id, dispatch_core_config=ttnn.device.DispatchCoreConfig())

# 32*8是一个core应该处理的量
# 如果大于32*8*8，应该效仿attention.py，将torch_input_tensor_a reshape到batch维度
# TODO: K值有问题，最后再改吧

# 定义张量尺寸
# [128*896*1152]是qwen2.5-0.5B的参数
M = 128
K = 896
N = 1152

# torch_input_tensor_a = torch.rand(1, 1, M, K, dtype=torch.float16)
# torch_input_tensor_a = torch_input_tensor_a * 2.0 - 1.0

# --- 周期定义 ---
MIN_VAL = -2.0
MAX_VAL = 2.0
STEP = 0.1  # 步长，决定周期的精细程度

batch_size = 1
# ---
num_steps = int((MAX_VAL - MIN_VAL) / STEP) + 1
period_tensor = torch.linspace(start=MIN_VAL, end=MAX_VAL, steps=num_steps, dtype=torch.float16)
total_elements = batch_size * M * K
period_length = period_tensor.numel()
num_repeats = (total_elements + period_length - 1) // period_length  # 向上取整
torch_input_tensor_a = period_tensor.repeat(num_repeats)[:total_elements]
torch_input_tensor_a = torch_input_tensor_a.view(batch_size, M, K)

input_tensor_a = ttnn.from_torch(
    torch_input_tensor_a,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
)
torch.set_printoptions(threshold=100000)

torch_gamma_tensor_before = torch.rand(K, dtype=torch.float16)
# torch_gamma_tensor_before = torch_input_tensor_a[0][0]
torch_gamma_tensor = torch_gamma_tensor_before.unsqueeze(0).unsqueeze(0).reshape([1, 1, K // 32, 32])
# print("torch_gamma_tensor")
# print(torch_gamma_tensor)
gamma_tensor = ttnn.from_torch(torch_gamma_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
# print("gamma_tensor")
# print(gamma_tensor)

# rmsnorm_torch = rmsnorm_no_weight(torch_input_tensor_a)
rmsnorm_torch = rmsnorm_with_weight(torch_input_tensor_a, torch_gamma_tensor_before)
# print("rmsnorm_torch")
# print(rmsnorm_torch)


rmsnorm = ttnn.rms_norm(input_tensor_a, weight=gamma_tensor, epsilon=1e-5)
torch_rmsnorm_output_tensor = ttnn.to_torch(rmsnorm)
# print("torch_rmsnorm_output_tensor[0]")
# print(torch_rmsnorm_output_tensor[0][0])
# print(torch_rmsnorm_output_tensor[0][1])


# torch_output_tensor = ttnn.to_torch(output_tensor)

torch_input_tensor_b = torch.rand(1, 1, K, N, dtype=torch.float16)
wqkv_mem_config = create_dram_sharded_mem_config(K, N // 1)
input_tensor_b = ttnn.from_torch(
    torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wqkv_mem_config
)


# output_tensor = ttnn.exp(input_tensor_a)

program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=1,
    per_core_N=5,
    out_block_h=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

# linear_norm 显示指定了out_block_h=1，因此out_subblock_h也只能=1.
program_config_fusenorm = ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=1,
    per_core_N=5,
    out_block_h=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

print("program_config_fusenorm=ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm")
compute_kernel_config = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,  # 变成true会出问题
    packer_l1_acc=True,
)

liner_output_tensor = ttnn.linear(
    rmsnorm, input_tensor_b, program_config=program_config, compute_kernel_config=compute_kernel_config
)
# print(liner_output_tensor)
torch_linear_output_tensor = ttnn.to_torch(liner_output_tensor)
# print("torch_linear_output_tensor")
# print(torch_linear_output_tensor[0])


torch_matmul_output_tensor = torch.matmul(rmsnorm_torch, torch_input_tensor_b)

# output_tensor = ttnn.exp(input_tensor_b)
# print(input_tensor_a)
# print(input_tensor_b)
# print(f"Memory config: {ttnn.get_memory_config(input_tensor_a)}")
# print(f"Memory config: {ttnn.get_memory_config(input_tensor_b)}")


linear_norm_output_tensor = ttnn.linear_norm(
    input_tensor_a,
    input_tensor_b,
    gamma=gamma_tensor,
    epsilon=1e-5,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    program_config=program_config_fusenorm,
    # compute_kernel_config=compute_kernel_config,
)
print(f"Input B allocated: {linear_norm_output_tensor.is_allocated()}")
print(f"Input A device: {linear_norm_output_tensor.device()}")
# print(linear_norm_output_tensor)
torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)

# torch.set_printoptions(threshold=10000)
# print(torch_matmul_output_tensor[0])
# print(torch_linear_norm_output_tensor[0])


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

print(f"me and torch PCC: {pcc_message}")

passing, pcc_message = comp_pcc(torch_linear_output_tensor, torch_linear_norm_output_tensor)

print(f"me and tt PCC: {pcc_message}")

passing, pcc_message = comp_pcc(torch_linear_output_tensor, torch_matmul_output_tensor)

print(f"tt and torch PCC: {pcc_message}")
