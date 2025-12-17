// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "pad_tile.hpp"
#include "ckernel.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

#include "debug/dprint.h"

// 与ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp进行融合
void kernel_main() {
    // DPRINT << "Reader kernel begin: kernel_main" << ENDL();

    uint32_t rt_args_idx = 0;
    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in0 mcast args
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args
    const uint32_t last_block_h = get_arg_val<uint32_t>(rt_args_idx++);
    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    /*
    norm额外的参数
    */
    // uint32_t src_addr = get_arg_val<uint32_t>(rt_args_idx++);       //输入张量在DRAM的地址
    // uint32_t NCHt = get_arg_val<uint32_t>(rt_args_idx++);           //单core需要处理的 tile 行数
    // uint32_t Wt = get_arg_val<uint32_t>(rt_args_idx++);             //宽度维度的 tile 数量
    // uint32_t tile_offset = get_arg_val<uint32_t>(rt_args_idx++);    // 当前 core 处理的起始 tile 行
    // Generate constant tiles for layernorm compute

    {
        constexpr uint32_t cb_in_12 = tt::CBIndex::c_12;

        uint32_t scaler = get_arg_val<uint32_t>(rt_args_idx++);
        generate_reduce_scaler(cb_in_12, scaler);
    }
    constexpr uint32_t eps_cb_id = 11;
    const uint32_t eps = get_arg_val<uint32_t>(rt_args_idx++);
    generate_bcast_col_scalar(eps_cb_id, eps);
    uint32_t gamma_addr = get_arg_val<uint32_t>(rt_args_idx++);
    // uint32_t beta_addr = get_arg_val<uint32_t>(rt_args_idx++);
    // DPRINT << "=== eps_cb_id after=== " << ENDL();
    // DPRINT << TileSlice(eps_cb_id, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true,
    // false)
    //        << ENDL();

    // COMPILE TIME ARGS
    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_next_inner_dim_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_h_dim_block_stride = get_compile_time_arg_val(3);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t in0_last_ktile_w = get_compile_time_arg_val(7);

    constexpr bool extract_shard_sub_blocks = (bool)get_compile_time_arg_val(8);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(10);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(12);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(13);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(16);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(17);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(18);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(19);



    // sparsity args
    // 控制稀疏矩阵的参数，先不管
    constexpr uint32_t batchB = get_compile_time_arg_val(20);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(21);
    // Boolean that is set when input A is sparse. If set, both input A and B are assumed to be sparse.
    // Based on the sparsity tensor, the corresponding batch in input A and B are skipped.
    constexpr bool bcast_A = (bool)get_compile_time_arg_val(22);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(23);

    //普通的ttnn.linear不会调用
    constexpr bool fuse_op = (bool)get_compile_time_arg_val(24);

    constexpr auto in0_args = TensorAccessorArgs<25>();
    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<sparsity_args.next_compile_time_args_offset()>();
    constexpr uint32_t stick_size = get_compile_time_arg_val(gamma_args.next_compile_time_args_offset());

    // Reader will use this CB to pass the number of non-zero (nnz) entries in the sparsity tensor.
    constexpr uint32_t nnz_cb_id = tt::CBIndex::c_25;

    // 0 is used to specify "INVALID" state, i.e. when the multicasted data has not been received by the receiver.
    // 0x1 is used to specify "VALID" state, i.e. when the batch is valid.
    // 0x2 is used to specify "IGNORE_BATCH" state, i.e. when the batch is not valid.
    constexpr uint32_t IGNORE_BATCH = 0x2;

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    MatmulOpReceiver fused_op_receiver;
    // if constexpr (fuse_op) {
    //     fused_op_receiver = MatmulOpReceiver(
    //         true, /* wait_for_op_signal */
    //         rt_args_idx,
    //         num_blocks_inner_dim,
    //         in0_block_w /* tiles_per_block (in the same dimension as tensor slice) */
    //     );
    // }

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_gamma = 10;
    // constexpr auto cb_id_gamma = tt::CBIndex::c_10;
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t in0_block_size_bytes = in0_block_num_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t one_tile = 1;

    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, stick_size);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);

    constexpr auto cb_norm_output = tt::CBIndex::c_16;

    // // sparsity accessor
    // constexpr uint32_t cb_id_sparsity = tt::CBIndex::c_6;
    // const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);


    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_receiver_semaphore_addr);
    // DPRINT << "in0_mcast_dest_noc_start_x:" << in0_mcast_dest_noc_start_x << ENDL();
    // DPRINT << "in0_mcast_dest_noc_start_y:" << in0_mcast_dest_noc_start_y << ENDL();
    // DPRINT << "in0_mcast_dest_noc_end_x:" << in0_mcast_dest_noc_end_x << ENDL();
    // DPRINT << "in0_mcast_dest_noc_end_y:" << in0_mcast_dest_noc_end_y << ENDL();

    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

    uint32_t l1_write_addr_sparsity = 0;
    // if constexpr (batchB > 0) {
    //     cb_reserve_back(cb_id_sparsity, 1);
    //     l1_write_addr_sparsity = get_write_ptr(cb_id_sparsity);
    // }

    for (uint32_t b = 0; b < batch; ++b) {
        // if constexpr (batchB > 0) {
        //     noc_async_read_page(b, s_sparsity, l1_write_addr_sparsity);
        //     noc_async_read_barrier();
        // }

        for (uint32_t bB = 0; bB < batchB_lim; ++bB) {
            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim;
                 ++bh) {  // pre_core_M（这就是将之前一次性读取多个变为一次性读取一行）
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {  // 1
                    uint32_t in0_tensor_current_inner_dim_block_start_tile_id = in0_tensor_current_h_dim_block_tile_id;
                    // 对应 for (uint32_t wt = 0; wt < Wt; wt += blk)
                    for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                        cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                        uint32_t in0_start_address =
                            l1_write_addr_in0;  // copy start address of block, to be used for mcasting

                        // Copy in0 block into CB, as the default kernel
                        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_inner_dim_block_start_tile_id;
                        for (uint32_t h = 0; h < in0_block_h; ++h) {  // 显示指定为1
                            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                            // 对应 for (uint32_t r = 0; r < blk; r++)
                            for (uint32_t w = 0; w < in0_block_w; ++w) {
                                if (bh < num_blocks_h_dim - 1 || h < last_block_h) {
                                    // DPRINT << "noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);" << "
                                    // bh=" << bh
                                    // << ",  bw=" << bw << ",  block=" << block << ",  h=" << h << ", w=" << w <<
                                    // ENDL();
                                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                                }

                                // Zero out padded regions for the very last tile
                                // 最后一个block的零填充
                                if constexpr (in0_last_ktile_w > 0) {
                                    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
                                        noc_async_read_barrier();
                                        const DataFormat in0_data_format = get_dataformat(cb_id_in0);
                                        pad_last_ktile<in0_data_format, in0_last_ktile_w>(l1_write_addr_in0);
                                    }
                                }

                                l1_write_addr_in0 += in0_single_tile_size_bytes;  // 字节数
                                in0_tensor_tile_id += in0_tensor_stride_w;        // 索引id
                            }
                            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
                        }
                        in0_tensor_current_inner_dim_block_start_tile_id += in0_tensor_next_inner_dim_block_stride;

                        // Barrier! make sure the reads are done
                        noc_async_read_barrier();

#ifndef SKIP_MCAST
                        // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                        // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back to
                        // zero for the next block
                        noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                        noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;

                        // num_dests must not include source, since we are NOT really doing a local copy!
                        noc_async_write_multicast(
                            in0_start_address,
                            in0_multicast_data_addr,
                            in0_block_size_bytes,
                            in0_mcast_num_cores,
                            true);

                        noc_semaphore_set_multicast(
                            in0_mcast_receiver_semaphore_addr,
                            in0_mcast_receiver_semaphore_noc_addr,
                            in0_mcast_num_cores);
#endif  // SKIP_MCAST

                        cb_push_back(cb_id_in0, in0_block_num_tiles);
                        // DPRINT << "cb_push_back(cb_id_in0, in0_block_num_tiles); END END END" << ENDL();
                    }
#ifdef FUSE_GAMMA
                    // 1.读取gamma会导致卡住 2.广播会导致除了(0,0)的其他core更早卡住
                    // 读取gamma
                    // 对应 for (uint32_t wt = 0; wt < Wt; wt += blk)
                    if (bh == 0) {
                        for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                            cb_reserve_back(cb_id_gamma, in0_block_num_tiles);
                            // DPRINT << "cb_reserve_back(cb_id_gamma, in0_block_num_tiles); END END END" << ENDL();

                            uint32_t l1_write_addr = get_write_ptr(cb_id_gamma);
                            uint32_t gamma_start_address =
                                l1_write_addr;  // copy start address of block, to be used for mcasting

                            // 对应 for (uint32_t r = 0; r < blk; r++)
                            for (uint32_t w = 0; w < in0_block_w; ++w) {
                                uint64_t gamma_noc_addr = get_noc_addr(block + w, addrg);
                                noc_async_read(gamma_noc_addr, l1_write_addr, 64);
                                // DPRINT << "noc_async_read(gamma_noc_addr, l1_write_addr, 64); END END END" << ENDL();
                                gamma_noc_addr = get_noc_addr(l1_write_addr + 32);
                                noc_async_read_barrier();
                                noc_async_read(gamma_noc_addr, l1_write_addr + 512, 32);
                                // DPRINT << "noc_async_read(gamma_noc_addr, l1_write_addr + 512, 32); END END END"
                                //        << ENDL();
                                l1_write_addr += gamma_tile_bytes;
                            }

                            // Barrier! make sure the reads are done
                            noc_async_read_barrier();

#ifndef SKIP_MCAST
                            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr
                            // (i.e. its value should be in0_mcast_num_dests), then reset the semaphore_addr value back
                            // to zero for the next block
                            noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                            noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                            // Now we have the block in the CB address, we can mcast to dests!
                            uint64_t in0_multicast_data_addr = in0_multicast_data_noc | gamma_start_address;

                            // num_dests must not include source, since we are NOT really doing a local copy!
                            noc_async_write_multicast(
                                gamma_start_address,
                                in0_multicast_data_addr,
                                in0_block_size_bytes,
                                in0_mcast_num_cores,
                                true);

                            noc_semaphore_set_multicast(
                                in0_mcast_receiver_semaphore_addr,
                                in0_mcast_receiver_semaphore_noc_addr,
                                in0_mcast_num_cores);
#endif  // SKIP_MCAST

                            // DPRINT << "cb_push_back(cb_id_gamma, in0_block_num_tiles); " << ENDL();
                            cb_push_back(cb_id_gamma, in0_block_num_tiles);
                            // DPRINT << "cb_push_back(cb_id_gamma, in0_block_num_tiles); bh: " << bh
                            //        << " block: " << block << ENDL();
                        }
                    }
#endif
                }
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
            }

            // if constexpr (!bcast_A) {
            //     in0_tensor_start_tile_id += MtKt;
            // }
        }

        // if constexpr (bcast_A) {
        //     in0_tensor_start_tile_id += MtKt;
        // }
    }
    noc_async_write_barrier();
    // DPRINT << "for (uint32_t b = 0; b < batch; ++b) END END END"  << ENDL();
}
