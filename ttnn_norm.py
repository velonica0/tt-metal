import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

#torch_input_tensor_a = torch.rand(32*8, 64*8, dtype=torch.float32)
torch_input_tensor_a = torch.ones(32*8, 64*8, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, 
    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
output_tensor = ttnn.exp(input_tensor_a)
torch_output_tensor = ttnn.to_torch(output_tensor)

torch_input_tensor_b = torch.rand(64*8, 32*8, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, 
    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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

program_config_fusenorm=ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=16,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=8,
    per_core_N=8,
    out_block_h=1,
    out_block_w=1,#不设置则取默认值per_core_N
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

# (8,16) * (16,8) 

# compute
# in0_block_h = out_block_h = 1 
# in1_block_w = out_block_w = 1
# (0) in0_block_w = in1_block_h = 8 
# (1) in0_num_subblocks = out_block_h / out_subblock_h = 1 / 1 = 1    //高度方向的subblock的个数
# (2) in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks = 8    //in0_block_w表示了宽度
# (3) in0_subblock_num_tiles = out_subblock_h * in0_block_w = 8    
# (4) in1_num_subblocks = (out_block_w / out_subblock_w) = 1
# (5) in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks = 8
# (6) in1_block_w = in1_per_core_w = out_subblock_w * in1_num_subblocks = 1
# (7) num_blocks_inner_dim = num_blocks = K / in0_block_w = 16 / 8 = 2
# (8) num_blocks_w_dim = out_num_blocks_x = in1_num_blocks_x = per_core_N / out_block_w = 8
# (9) num_blocks_h_dim = out_num_blocks_y = in0_num_blocks_y = per_core_M / out_block_h = 8
# (12) out_subblock_num_tiles = out_subblock_h * out_subblock_w = 1
# (13) batch = 32 * 8 = 256
# (14) out_block_num_tiles = out_block_tiles = out_block_h * out_block_w = 1 * 1 = 1

# in1_sender_writer_compile_time_args
# (0)in1_tensor_stride_w=1
# (1)in1_tensor_stride_h=N=8
# (2)in1_tensor_next_block_stride=in0_block_w * N=8*8
# (3)in1_tensor_next_w_dim_block_stride=in1_block_w=1
# (4)in1_block_w =1              
# (5)in1_block_h=in0_block_w =8              
# (6)in1_block_num_tiles=in1_block_w * in0_block_w=1*8=8
# (7)num_blocks_inner_dim=num_blocks=K / in0_block_w
# (8)num_blocks_w_dim=out_num_blocks_x=per_core_N / out_block_w = 8
# (9)num_blocks_h_dim=out_num_blocks_y=per_core_M / out_block_h = 8
# (10)in1_mcast_sender_semaphore_id
# (11)in1_mcast_receiver_semaphore_id
# (12)in1_mcast_num_dests=num_blocks_y - 1
# (13)in1_mcast_num_cores=num_blocks_y - 1 
# (14)KtNt=K * N =16 * 8 = 128         
# (15)B            
# (16)bcast_batch  
# (17)0  
# (18)0  

#liner_output_tensor = ttnn.linear(input_tensor_a, input_tensor_b, program_config=program_config)
linear_norm_output_tensor = ttnn.linear_norm(input_tensor_a, input_tensor_b, gamma=None, epsilon=1e-5,program_config=program_config_fusenorm)
# torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor)


ttnn.close_device(device)
