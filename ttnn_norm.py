import torch
import ttnn
import numpy as np

device_id = 0
device = ttnn.open_device(device_id=device_id)

M = 32 * 8
K = 32 * 16
N = 32 * 1

row_values = torch.arange(1, M + 1, dtype=torch.float32).unsqueeze(1)
torch_input_tensor_a = row_values.expand(-1, K)
# torch_input_tensor_a = torch.rand(M, K, dtype=torch.float32)
# torch_input_tensor_a = torch.ones(32*8*1, 32*16, dtype=torch.float32)
# torch_input_tensor_a = torch.full((M, K), fill_value=1.0, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, 
    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
output_tensor = ttnn.exp(input_tensor_a)
torch_output_tensor = ttnn.to_torch(output_tensor)



# start_vals = torch.arange(1, K + 1, dtype=torch.int64).unsqueeze(1)
# col_offsets = torch.arange(0, N, dtype=torch.int64).unsqueeze(0)
# torch_input_tensor_b = start_vals + col_offsets
# torch_input_tensor_b = torch.rand(K, N, dtype=torch.float32)
# torch_input_tensor_b = torch.ones(32*1, 32*1, dtype=torch.float32)
torch_input_tensor_b = torch.full((K, N), fill_value=2.0, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, 
    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=8,
    per_core_N=1,
    out_block_h=1,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

program_config_fusenorm=ttnn.MatmulMultiCoreReuseMultiCastProgramConfigFuseNorm(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
    out_subblock_h=1,  # Must be divisible by per_core_M
    out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
    per_core_M=8,
    per_core_N=1,
    out_block_h=1,
    out_block_w=1,#不设置则取默认值per_core_N，需要适配非1场景
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=False,
)

compute_kernel_config = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
# (8,16) * (16,8) 

# compute
# in0_block_h = out_block_h = 1 
# in1_block_w = out_block_w = 1
# (0) in0_block_w = in1_block_h = 8 
# (1) in0_num_subblocks = out_block_h / out_subblock_h = 1 / 1 = 1    //高度方向的subblock的个数
# (2) in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks = 8    //in0_block_w表示了宽度
#                         = in0_block_w * out_block_h
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

# in0_sender_compile_time_args
# (0) in0_tensor_stride_w = 1
# (1) in0_tensor_stride_h = K = 16
# (2) in0_tensor_next_inner_dim_block_stride = in0_block_w = 8 
# (3) in0_tensor_next_h_dim_block_stride = K*in0_block_h = 16
# (4) in0_block_w = 8
# (5) in0_block_h = 1
# (6) in0_block_num_tiles = 8
# (7) in0_last_ktile_w
# (8) extract_shard_sub_blocks = false (not used for interleaved)
# (9) shard_width_in_tiles = in0_shard_width_in_tiles (not used for interleaved)
# (10) shard_height_in_tiles = in0_shard_height_in_tiles (not used for interleaved)
# (11) num_blocks = K / in0_block_w = 16 / 8 = 2
# (12) out_num_blocks_x = in1_num_blocks_x = per_core_N / out_block_w = 8
# (13) out_num_blocks_y = in0_num_blocks_y = per_core_M / out_block_h = 8
# (14) in0_mcast_sender_semaphore_id
# (15) in0_mcast_receiver_semaphore_id
# (16) in0_mcast_num_dests = (num_blocks_x - 1) 
# (17) in0_mcast_num_cores = (num_blocks_x - 1) 
# (18) MtKt = M*K = 8 * 16 = 128  
# (19) batch = B   
# (20) batchB = 0    
# (21) sparsity_pagesize = 0 (placeholder since sparsity not used in this case)
# (22) bcast_A = true  
# (23) get_batch_from_reader = false

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

# # 1. 关闭 NumPy 打印截断（关键：让 numpy 输出全部内容）
# np.set_printoptions(
#     threshold=np.inf,  # 取消打印阈值（不截断）
#     linewidth=np.inf,  # 取消行宽限制（避免换行混乱）
#     suppress=True      # 禁用科学计数法（可选，按需开启）
# )

# # 2. （可选）关闭 PyTorch 张量打印截断（直接打印 torch 张量时用）
# torch.set_printoptions(
#     threshold=torch.inf,  # 取消张量打印阈值
#     linewidth=10000       # 行宽设足够大，避免换行
# )


linear_norm_output_tensor = ttnn.linear_norm(
    input_tensor_a, 
    input_tensor_b, 
    gamma=None, 
    epsilon=1e-5,
    program_config=program_config_fusenorm,
    compute_kernel_config=compute_kernel_config)

norm_output_tensor = ttnn.rms_norm(input_tensor_a, epsilon=1e-5, compute_kernel_config=compute_kernel_config)
liner_output_tensor = ttnn.linear(norm_output_tensor, input_tensor_b, program_config=program_config)

# torch_norm_output_tensor = ttnn.to_torch(norm_output_tensor).to(dtype=torch.float32)
# print(torch_norm_output_tensor.cpu().numpy())
torch_liner_output_tensor = ttnn.to_torch(liner_output_tensor).to(dtype=torch.float32)
# print(torch_liner_output_tensor.cpu().numpy())

torch_linear_norm_output_tensor = ttnn.to_torch(linear_norm_output_tensor).to(dtype=torch.float32)
# print(torch_linear_norm_output_tensor.cpu().numpy())


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
    
# --- 示例用法 ---# 假设我们有两个 3x4 的二维 Tensor# 计算并打印结果
# passing, pcc_message = comp_pcc(torch_matmul_output_tensor, torch_linear_norm_output_tensor)
# print(f"me and torch PCC: {pcc_message}")
passing, pcc_message = comp_pcc(torch_liner_output_tensor, torch_linear_norm_output_tensor)
print(f"me and tt PCC: {pcc_message}")
# passing, pcc_message = comp_pcc(torch_linear_output_tensor, torch_matmul_output_tensor)
# print(f"tt and torch PCC: {pcc_message}")