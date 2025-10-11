// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);

    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr uint32_t is_causal = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);

    const uint32_t core_id = get_arg_val<uint32_t>(0);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);
    // const uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(7);
    const uint32_t num_phases = get_arg_val<uint32_t>(7);
    const uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(8);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(9);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    uint32_t chunked_q_chunk_offset = 0;
    mm_init(cb_q_in, cb_k_in, cb_out);

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2;
        }

        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                    uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
                uint32_t q_chunk_div_2 = q_chunks_per_core / 2;
                if (q_iter < q_chunk_div_2) {  // bottom half
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunk_div_2;  // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
#else
                q_chunk = local_q_start + q_iter;
#endif

                // Get Q chunk
                if constexpr (is_chunked) {
                    q_chunk = chunked_q_chunk_offset + q_chunk;
                }
                uint32_t q_low_idx =
                    q_chunk * Sq_chunk_t;  // This is the sequence index of the first tile of this chunk
                uint32_t q_high_idx;
                if constexpr (is_causal) {
                    q_high_idx = q_low_idx + Sq_chunk_t;
                } else {
                    q_high_idx = Skt;
                }

                // Set up ping pong buffers
                uint32_t alias_prev_sum = cb_sum_A;
                uint32_t alias_cur_sum = cb_sum_B;
                uint32_t alias_prev_max = cb_max_A;
                uint32_t alias_cur_max = cb_max_B;
                uint32_t alias_mm2_prev_out = cb_out_im_A;
                uint32_t alias_mm2_cur_out = cb_out_im_B;

                // 第二个同步点：等待Query分块数据准备就绪，matmul_blocks内部等待Key
                cb_wait_front(cb_q_in, q_chunk_tiles);  // 等待Query分块数据准备就绪
                // loop while k_low < q_high
                // 循环处理所有Key分块，直到k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;     // 当前Key分块的起始位置索引
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;  // 当前Key分块的结束位置索引

                    /* QK = Q_CHUNK @ K_CHUNK */
                    /* QK = Q分块 @ K分块 - 计算注意力分数矩阵 */
                    pack_reconfig_data_format(cb_qk_im);  // 重新配置QK输出缓冲区的数据格式
                    matmul_blocks(                        // 执行矩阵乘法：Q分块 × K分块 = QK注意力分数
                        cb_q_in,                          // Query输入缓冲区
                        cb_k_in,                          // Key输入缓冲区
                        cb_qk_im,                         // QK输出缓冲区
                        Sq_chunk_t,                       // Query分块序列长度
                        Sk_chunk_t,                       // Key分块序列长度
                        DHt,                              // 头维度
                        qk_num_blocks,                    // QK矩阵乘法的块数量
                        qk_in0_num_subblocks,             // Q分块的子块数量
                        qk_in1_num_subblocks,             // K分块的子块数量
                        qk_in0_block_w,                   // Q分块的块宽度
                        qk_subblock_h,                    // QK子块高度
                        qk_subblock_w,                    // QK子块宽度
                        true /*transpose*/);              // 对K进行转置

                    /**
                     * Note
                     * Typically, scores is multiplied by a scalar here. We employed an optimization
                     * where we fuse the scaling into exp both in exp(x - max) and exp(prev_max - cur_max).
                     * This gives us scaling for free on the performance-critical exp(x - max) computation.
                     */
                    /**
                     * 注意
                     * 通常，这里会对分数乘以一个标量。我们采用了一种优化策略，
                     * 将缩放因子融合到exp计算中，包括exp(x - max)和exp(prev_max - cur_max)。
                     * 这使得我们在性能关键的exp(x - max)计算中免费获得缩放效果。
                     */

                    // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                    // Q-range = [q_low, q_high)
                    // K-range = [k_low, k_high)
                    // does_overlap = not (q_low >= k_high or k_low >= q_high)
                    // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                    // 由于q_chunk_size和k_chunk_size可能不同，找到对角线变得更加困难
                    // Q范围 = [q_low, q_high)
                    // K范围 = [k_low, k_high)
                    // 是否重叠 = not (q_low >= k_high or k_low >= q_high)
                    // 由于循环边界，我们永远不会出现k_low >= q_high的情况。可以简化这个条件检查
                    if constexpr (is_causal) {             // 如果是因果注意力（GPT类型）
                        if (!(q_low_idx >= k_high_idx)) {  // 如果Q和K分块有重叠
                            /* QK += MASK */
                            /* QK += 掩码 - 添加因果掩码 */
                            reconfig_data_format(cb_qk_im, cb_mask_in);               // 重新配置数据格式以匹配掩码
                            add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);  // 就地添加掩码到QK矩阵
                        }
                    } else if constexpr (use_provided_mask) {  // 如果使用提供的掩码
                        /* QK += MASK */
                        /* QK += 掩码 - 添加提供的掩码 */
                        reconfig_data_format(cb_qk_im, cb_mask_in);               // 重新配置数据格式以匹配掩码
                        add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);  // 就地添加掩码到QK矩阵
                    } else if constexpr (use_padded_mask) {                       // 如果使用填充掩码
                        // only uses mask on the last K chunk if it exists at all
                        // 只在最后一个K分块上使用掩码（如果存在的话）
                        if (k_chunk == k_num_chunks - 1) {  // 如果是最后一个K分块
                            /* QK += MASK */
                            /* QK += 掩码 - 添加填充掩码 */
                            reconfig_data_format(cb_qk_im, cb_mask_in);               // 重新配置数据格式以匹配掩码
                            add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);  // 就地添加掩码到QK矩阵
                        }
                    }

                    /**
                     * reduce_c can perform both reduce_max and eltwise max with previous result.
                     * if do_eltwise_max:
                     *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                     * else:
                     *  cur_max = max(qk, dim=-1)
                     */
                    /**
                     * reduce_c可以执行reduce_max和与前一次结果的逐元素最大值操作。
                     * 如果do_eltwise_max为真：
                     *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                     * 否则：
                     *  cur_max = max(qk, dim=-1)
                     */
                    reconfig_data_format(cb_qk_im, cb_identity_scale_in);  // 重新配置数据格式以匹配缩放因子
                    reduce_c<                                              // 执行行方向的最大值归约
                        PoolType::MAX,                                     // 使用MAX池化类型
                        ReduceDim::REDUCE_ROW,                             // 在行方向进行归约
                        cb_qk_im,                                          // QK输入缓冲区
                        cb_identity_scale_in,                              // 缩放因子缓冲区
                        Sq_chunk_t,                                        // Query分块序列长度
                        Sk_chunk_t>(
                        alias_cur_max,
                        alias_prev_max,
                        k_chunk > 0);  // 输出到当前最大值，输入前一次最大值，是否进行逐元素最大值

                    /**
                     * sub_exp fuses a few operations.
                     * In-place it performs `QK = exp((QK - cur_max) * scale)`
                     *
                     * It also partially performs reduce_sum on the output using L1 accumulation.
                     * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
                     *
                     * Partial reduce_sum is used to push the final row_reduction within a tile
                     * outside of the loop over K chunks.
                     */
                    /**
                     * sub_exp融合了几个操作。
                     * 就地执行 `QK = exp((QK - cur_max) * scale)`
                     *
                     * 它还使用L1累积对输出进行部分reduce_sum操作。
                     * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
                     *
                     * 部分reduce_sum用于将tile内的最终行归约推到K分块循环之外。
                     */
                    sub_exp_block_bcast_cols_inplace<
                        cb_qk_im,
                        Sq_chunk_t,
                        Sk_chunk_t,
                        scale_fp32>(  // 执行缩放指数操作并广播列
                        alias_cur_max,
                        alias_cur_sum);  // 使用当前最大值和当前和

                    cb_wait_front(cb_qk_im, qk_chunk_tiles);  // 等待QK注意力分数矩阵准备就绪
                    /* OUT_IM = QK @ V_CHUNK */
                    /* OUT_IM = QK @ V分块 - 计算注意力输出 */
                    matmul_blocks(              // 执行矩阵乘法：注意力分数 × Value分块 = 输出
                        cb_qk_im,               // QK注意力分数输入缓冲区
                        cb_v_in,                // Value输入缓冲区
                        alias_mm2_cur_out,      // 当前输出缓冲区
                        Sq_chunk_t,             // Query分块序列长度
                        vDHt,                   // Value头维度
                        Sk_chunk_t,             // Key分块序列长度
                        out_num_blocks,         // 输出矩阵乘法的块数量
                        out_in0_num_subblocks,  // QK分块的子块数量
                        out_in1_num_subblocks,  // V分块的子块数量
                        out_in0_block_w,        // QK分块的块宽度
                        out_subblock_h,         // 输出子块高度
                        out_subblock_w,         // 输出子块宽度
                        false /*transpose*/);   // 不对V进行转置

                    cb_pop_front(cb_qk_im, qk_chunk_tiles);               // 释放QK注意力分数矩阵缓冲区
                    reconfig_data_format(alias_prev_max, alias_cur_max);  // 重新配置最大值缓冲区的数据格式

                    /* OUT_ACC += OUT_IM */
                    /* 输出累积 += 输出中间结果 - 累积不同K分块的结果 */
                    if (k_chunk > 0) {  // 如果不是第一个K分块，需要累积结果
                        /**
                         * cb_exp_max_diff = torch.exp((cb_prev_max - cb_cur_max) * scale)
                         * Scale is fused into exp again since max is the max of unscaled scores.
                         */
                        /**
                         * cb_exp_max_diff = torch.exp((cb_prev_max - cb_cur_max) * scale)
                         * 由于max是未缩放分数的最大值，缩放因子再次融合到exp中。
                         */

                        sub_exp_block<scale_fp32>(
                            alias_prev_max,
                            alias_cur_max,
                            cb_exp_max_diff,
                            Sq_chunk_t);                           // 计算exp((prev_max - cur_max) * scale)
                        cb_pop_front(alias_prev_max, Sq_chunk_t);  // 释放前一次最大值缓冲区

                        /**
                         * cb_prev_sum *= cb_exp_max_diff
                         * This is a bcast_cols since max_diff is a column vector and prev_sum is a partial
                         * reduction, containing the sum of tiles in dim=-1 of QK.
                         */
                        /**
                         * cb_prev_sum *= cb_exp_max_diff
                         * 这是一个列广播操作，因为max_diff是列向量，prev_sum是部分归约结果，
                         * 包含QK在dim=-1方向上的tile和。
                         */
                        mul_tiles_bcast_cols_inplace(
                            alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);  // 前一次和乘以最大值差异
                        /* cb_cur_sum += cb_prev_sum */
                        /* cb_cur_sum += cb_prev_sum - 累积和 */
                        add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);  // 当前和加上前一次和

                        /**
                         * alias_mm2_cur_out += alias_mm2_prev_out * cb_exp_max_diff
                         * This uses L1 accumulation to accumulate onto mm2_cur_out.
                         */
                        /**
                         * alias_mm2_cur_out += alias_mm2_prev_out * cb_exp_max_diff
                         * 这使用L1累积来累积到mm2_cur_out上。
                         */
                        mul_block_bcast_cols<Sq_chunk_t, vDHt>(  // 执行块乘法并广播列
                            alias_mm2_prev_out,
                            cb_exp_max_diff,
                            alias_mm2_cur_out,
                            true);  // 前一次输出乘以最大值差异，累积到当前输出
                    }

                    // Swap CB handles to prepare for next iteration
                    // 交换CB句柄以为下一次迭代做准备
                    std::swap(alias_prev_sum, alias_cur_sum);          // 交换和缓冲区
                    std::swap(alias_mm2_prev_out, alias_mm2_cur_out);  // 交换输出缓冲区
                    std::swap(alias_prev_max, alias_cur_max);          // 交换最大值缓冲区
                }

                /**
                 * Performs final row-reduction on the partial sum.
                 */
                /**
                 * 对部分和执行最终的行归约。
                 */
                matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);  // 执行矩阵乘法归约，完成softmax分母计算
                /* cb_cur_sum = 1.0 / cb_cur_sum */
                /* cb_cur_sum = 1.0 / cb_cur_sum - 计算softmax归一化因子 */
                recip_block_inplace(alias_prev_sum, Sq_chunk_t);  // 计算倒数，得到softmax归一化因子

                /* cb_out_accumulate_im *= cb_cur_sum */
                /* cb_out_accumulate_im *= cb_cur_sum - 应用softmax归一化 */
                pack_reconfig_data_format(cb_out);  // 重新配置输出缓冲区的数据格式
                mul_block_bcast_cols<Sq_chunk_t, vDHt>(
                    alias_mm2_prev_out, alias_prev_sum, cb_out, false);  // 输出乘以归一化因子，得到最终结果

                cb_pop_front(cb_q_in, q_chunk_tiles);  // 释放Query分块缓冲区
                // free up cb_prev_max after K chunks
                // 在K分块处理完成后释放cb_prev_max
                cb_pop_front(alias_prev_max, Sq_chunk_t);  // 释放最大值缓冲区
                }
            }
        }
    }
}
}  // namespace NAMESPACE
