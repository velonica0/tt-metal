// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles   表示每次从 circular buffer 读取输入 A 时,沿 K 维度读取多少个 tiles。这是矩阵乘法内积计算的累加维度。
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);        // outer row block size (in inner row blocks)  in0_num_subblocks = per_core_M / out_subblock_h 表示每个核心的输出在 M 维度上被分成多少个 subblock。
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_h*in0_block_w*in0_num_subblocks;   这是输入 A 在一次完整迭代(block < num_blocks)中需要的所有 tiles。
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);   // out_subblock_h*in0_block_w  这是计算一个输出 subblock 时需要从输入 A 读取的 tiles 数量。
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);        // outer column block size (in inner column blocks)        表示每个核心的输出在 N 维度上被分成多少个 subblock。
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);      // out_subblock_w*in0_block_w* in1_num_subblocks;  这是输入 B 在一次完整迭代中需要的所有 tiles。
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);           // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);               // outer inner dim (in inner dim blocks)   num_blocks = K / in0_block_w    表示需要迭代多少次才能完成 K 维度的累加。
    uint32_t out_subblock_h = get_compile_time_arg_val(8);           // inner row block size in tiles 输出subblock的高度  由SUBBLOCK_HW_CHOICES决定
    uint32_t out_subblock_w = get_compile_time_arg_val(9);           // inner column block size in tiles 输出subblock的宽度  由SUBBLOCK_HW_CHOICES决定
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);  // out_subblock_h * out_subblock_w;
    uint32_t batch = get_compile_time_arg_val(11);                   // batch dim

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);
            // L1一次性接收的数量，block变量的迭代是在reader kernel
            cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
            cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
            // subblock间的迭代
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    // CB24充当临时累加区域，防止上一轮的数据在DST计算时被覆盖
                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_1, tt::CBIndex::c_24);
                        cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(tt::CBIndex::c_24, i, i);
                        }
                        cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                        mm_init_short_with_dt(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);
                    }

                    // Compute output sub-block from in0_subblock x in1_subblock
                    // 计算单个subblock的矩阵乘积
                    // subblock只在输出矩阵上，因此是二维的，没有K维度
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                // reader kernel只会读取两个block，在这个地方按照两个方向进行分配

                                // in0_index_subblock_offset: 当前 subblock 的起始偏移量,每处理完一个 subblock 后增加 in0_subblock_num_tiles
                                // in0_index_h_offset: 当前行在 subblock 中的偏移量,每处理完一行后增加 in0_block_w
                                // inner_dim: 内部维度的索引 (0 到 in0_block_w-1),对应矩阵乘法中的 K 维度
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                // in1_index_subblock_offset: 当前 subblock 的起始偏移量,每处理完一个 subblock 后增加 out_subblock_w
                                // in1_index_inner_dim_offset: 内部维度的偏移量,每处理一个 inner_dim 后增加 in1_per_core_w,用于在 K 维度上移动 
                                // w: 列索引 (0 到 out_subblock_w-1),对应矩阵乘法中的 N 维度
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(
                                    tt::CBIndex::c_0,
                                    tt::CBIndex::c_1,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    false /* transpose */);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        // Pack out to output buffer
                        cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_16);
                        }
                        cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                    } else {
                        // Wait for tiles in output buffer to be written out since interm and output share memory
                        if (block == 0) {
                            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Move partial result to interm buffer
                        cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, tt::CBIndex::c_24);
                        }
                        cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
            cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
        }
    }
}
}  // namespace NAMESPACE
