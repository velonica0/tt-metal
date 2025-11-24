import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor_a = torch.rand(4, 7, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.exp(input_tensor_a)
torch_output_tensor = ttnn.to_torch(output_tensor)

torch_input_tensor_b = torch.rand(7, 1, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm(
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

print("MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm")

linear_norm_output_tensor = ttnn.linear_norm(input_tensor_a, input_tensor_b, gamma=None, epsilon=1e-5,program_config=program_config)
torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)


ttnn.close_device(device)
