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
torch_input_tensor_a = row_pattern.unsqueeze(0).expand(R, C).to(torch.float32)
# torch_input_tensor_a = torch.ones(R, C, dtype=torch.float32)

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
print(torch_linear_output_tensor)
print(torch_linear_output_tensor.shape[0])
print(torch_linear_output_tensor.shape[1])


output_tensor = ttnn.exp(input_tensor_b)
linear_norm_output_tensor = ttnn.linear_norm(
    input_tensor_a, input_tensor_b, gamma=None, epsilon=1e-5, program_config=program_config_fusenorm
)
torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)

torch.set_printoptions(threshold=10000)
print(torch_linear_norm_output_tensor[0])


ttnn.close_device(device)
