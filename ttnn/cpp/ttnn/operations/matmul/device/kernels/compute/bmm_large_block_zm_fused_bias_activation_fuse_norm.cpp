// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "mod_div_lib.h"

#ifdef FUSE_BIAS
#include "compute_kernel_api/bcast.h"
#endif

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "debug/dprint.h"


#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "dprint_tensix.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.

namespace NAMESPACE {

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    // Reconfigure input
    // 将 mm_partials_cb_id 配置为 srcA 的数据源，copy_tile_to_dst_init_short_with_dt底层调用reconfig_data_format_srca
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure srcA back
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}


void MAIN {

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);     // outer inner dim (in inner dim blocks) block在K的个数  =  num_blocks
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);         // outer inner dim (in inner dim blocks) block在W的个数
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);         // outer inner dim (in inner dim blocks) block在H的个数
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(16);

    constexpr uint32_t do_gamma = get_compile_time_arg_val(17);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(18) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(19) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(20) == 1;

    // constexpr uint32_t Wt = get_compile_time_arg_val(21);                //宽度维度的 tile 数量
    // constexpr uint32_t blk = get_compile_time_arg_val(22);               //core单次处理多少个 tile

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;
    // Reader will use this CB to pass the number of non-zero (nnz) entries in the sparsity tensor.
    constexpr uint32_t nnz_cb_id = tt::CBIndex::c_25;
    volatile uint32_t* nnz_addr_ptr;
    /*rmsnorm的CB*/
    constexpr auto cb_scaler = tt::CBIndex::c_12;  // single tile 存储缩放因子 1/W,用于计算均值 它用于计算 E[x²],即对所有 x² 值求和后乘以 1/W
    constexpr auto cb_eps = tt::CBIndex::c_11;     // single tile 存储 epsilon 值,用于数值稳定性
    constexpr auto cb_out = tt::CBIndex::c_4;    // output
    constexpr auto cb_gamma = tt::CBIndex::c_10;
    constexpr uint32_t cb_xmm = in0_cb_id;  // x minus mean
    constexpr auto cb_ex2 = tt::CBIndex::c_19;     // 存储 E[x²],即 x 平方的均值
    constexpr auto cb_xmm2 = tt::CBIndex::c_20;    // 存储 x²,即每个元素的平方
    constexpr auto cb_ex2pe = tt::CBIndex::c_21;   // 存储 1/√(E[x²] + ε),即 RMS 归一化因子
    constexpr auto cb_fusion = tt::CBIndex::c_22;  // stream gamma/beta
    constexpr auto scaler0 = 0;
    constexpr auto cb_norm_output = tt::CBIndex::c_16;

    constexpr uint32_t untilize_mode_out_cb_id = out_cb_id;
    constexpr int cb_im_or_out = (do_gamma ) ? cb_fusion : cb_norm_output;

    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;

    // cb_wait_front(cb_scaler, 1);  // comes from the reader
    // cb_wait_front(cb_eps, 1);     // comes from the reader

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    DPRINT << "Compute kernel started" << ENDL();

    DPRINT_UNPACK(DPRINT << "me in0_num_subblocks" << in0_num_subblocks << ENDL());
    DPRINT_UNPACK(DPRINT << "me in1_num_subblocks" << in1_num_subblocks << ENDL());
    // DPRINT << "num_blocks_w_dim"<<num_blocks_w_dim<<ENDL();

    // mm_block_init(
    //     in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        // if constexpr (get_batch_from_reader) {
        //     // Check whether this batch is valid
        //     cb_wait_front(nnz_cb_id, 1);
        //     tensix_sync();
        //     cb_get_tile(nnz_cb_id, 0, &nnz_addr_ptr);
        //     // The first 4 entries have metadata, so we look at the 5th entry
        //     // for our value pushed from the reader.
        //     uint32_t nnz = nnz_addr_ptr[4];
        //     cb_release_tile(nnz_cb_id);
        //     cb_pop_front(nnz_cb_id, 1);

        //     if (nnz == 0) {
        //         continue;
        //     }
        // }

        // for (uint32_t ncht = 0; ncht < NCHt; ncht++)
        // ttnn.rmsnorm一个core处理一行tile，现在的话一个core处理num_blocks_h_dim行tile
        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {// 8

            /*
                rmsnorm的计算
            */
            constexpr int onetile = 1;
            constexpr int dst0 = 0;

            // X+Y
            reconfig_data_format(in0_cb_id, in0_cb_id);
            pack_reconfig_data_format(cb_xmm2);

            // DPRINT_UNPACK({
            //     DPRINT << "=== cb_xmm ===, bh:" << bh << ENDL();
            //     DPRINT
            //         << TileSlice(
            //                cb_xmm, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true,
            //                false)
            //         << ENDL();
            // })

            // 第一步：计算 x²
            // 结果存入 cb_xmm2
            mul_tiles_init(cb_xmm, cb_xmm);
            // DPRINT_UNPACK(DPRINT << "num_blocks_inner_dim" << num_blocks_inner_dim << ENDL());
            for (uint32_t wt = 0; wt < in0_block_w * num_blocks_inner_dim; wt += in0_block_w) {     // num_blocks_inner_dim = num_blocks = K / in0_block_w = K(Kt)
                // DPRINT_UNPACK(DPRINT << "cb_wait_front(cb_xmm, wt + in0_block_w);"<<", cb_xmm:"<<static_cast<uint32_t>(cb_xmm)<<", wt:"<<wt<<", in0_block_w:"<<in0_block_w << ENDL());
                cb_wait_front(cb_xmm, wt + in0_block_w);  // cumulative wait
                cb_reserve_back(cb_xmm2, in0_block_w);    // can probably use less space for this if we block
                ACQ();
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    // 第一次使用：计算x^2
                    mul_tiles(cb_xmm, cb_xmm, wt + wtr, wt + wtr, wtr);
                    // mul_tiles(cb_xmm, cb_col1, wt+wtr, wt+wtr, wtr);
                    pack_tile(wtr, cb_xmm2, wtr);
                    // if (wt == 0 && wtr == 0) {
                    //     DPRINT_UNPACK({
                    //         DPRINT << "=== cb_xmm(0,0) ===, bh:" << bh << ENDL();
                    //         DPRINT
                    //             << TileSlice<128>(
                    //                 cb_xmm, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                    //                 true, false)
                    //             << ENDL();
                    //     })
                    //     dprint_tensix_dest_reg(0);
                    // }
                }
                cb_push_back(cb_xmm2, in0_block_w);
                REL();
            }
            reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);

            DPRINT_UNPACK({
                DPRINT << "=== cb_xmm ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice<128>(
                           cb_xmm, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
                DPRINT << "=== cb_xmm2 ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice<128>(
                           cb_xmm2, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
            })

            // 第二步：计算E[x^2]归一化因子
            // 这个循环对 cb_xmm2 中的所有平方值进行归约求和,然后乘以 cb_scaler 得到均值,结果存入 cb_ex2。
            if constexpr (FLOAT32_DTYPE) {
                reconfig_data_format(cb_xmm2, cb_scaler);
            }
            cb_reserve_back(cb_ex2, 1);  // 在输出缓冲区 cb_ex2 中预留 1 个 tile 的空间用于存储结果
            reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                cb_xmm2, cb_scaler, cb_ex2);  // 行规约，每个tile(32*32)的结果为(1*32)
            ACQ();
            cb_wait_front(cb_xmm2, in0_block_w * num_blocks_inner_dim); //累积等待整行的所有 in0_block_w * num_blocks_inner_dim 个 tile 都准备好 这确保了归约操作可以访问完整的一行数据

            // 外层循环以 in0_block_w 为步长遍历所有 tile
            for (uint32_t wt = 0; wt < in0_block_w * num_blocks_inner_dim; wt += in0_block_w) {
                // reduce
                // 内层循环对每个 tile 调用 reduce_tile,将其累加到目标寄存器 dst0 中
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, wt + wtr, scaler0, dst0);
                }
            }

            cb_pop_front(cb_xmm2, in0_block_w * num_blocks_inner_dim);
            pack_tile(dst0, cb_ex2);
            reduce_uninit();
            REL();

            cb_push_back(cb_ex2, 1);
            cb_wait_front(cb_ex2, 1);

            DPRINT_UNPACK({
                DPRINT << "=== cb_ex2 ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice(
                           cb_ex2, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
            })

            /* Var(x) + eps
            * add epsilon E[(x-E[x])^2]+eps
            */
            if constexpr (FLOAT32_DTYPE) {
                reconfig_data_format(cb_ex2, cb_eps);
            }
            ACQ();
            add_tiles_init(cb_ex2, cb_eps);
            add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

            cb_reserve_back(cb_ex2pe, 1);  // 1
            rsqrt_tile_init<LEGACY_RSQRT>();
            rsqrt_tile<LEGACY_RSQRT>(dst0);
            pack_tile(dst0, cb_ex2pe);
            cb_push_back(cb_ex2pe, 1);
            REL();
            cb_pop_front(cb_ex2, 1);

            DPRINT_UNPACK({
                DPRINT << "=== cb_xmm ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice(
                           cb_xmm, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
                DPRINT << "=== cb_eps ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice(
                           cb_eps, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
                DPRINT << "=== cb_ex2pe ===, bh:" << bh << ENDL();
                DPRINT
                    << TileSlice(
                           cb_ex2pe, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1}, true, false)
                    << ENDL();
                DPRINT << "=== cb_scaler ===, bh:" << bh << ENDL();
                DPRINT << TileSlice(
                              cb_scaler,
                              0,
                              SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              true,
                              false)
                       << ENDL();
            })

            /* ln(x) * gamma + beta (gamma and beta are optional)
            * now xmm = (x-E[x])
            * we have 1.0/sqrt( E[(x-E[x])^2] + eps) in cb_ex2pe
            * just need to bcast_mul xmm with cb_ex2pe
            */
            // 第三步：分块归一化循环，这个循环将 cb_xmm 与归一化因子 cb_ex2pe (即 1/√(E[x²]+ε)) 相乘，结果存入
            // cb_fusion
            cb_wait_front(cb_ex2pe, 1);
            for (uint32_t wt = 0; wt < in0_block_w * num_blocks_inner_dim; wt += in0_block_w) {
                // if (ht == 1) UNPACK(( DPRINT << "wt_2=" << wt << " " ));
                // if (ht == 1) UNPACK(( DPRINT << "rem_2=" << rem << ENDL() ));
                reconfig_data_format(cb_xmm, cb_ex2pe);
                if constexpr (do_gamma == 0 ) {
                    pack_reconfig_data_format(cb_norm_output);
                } else {
                    pack_reconfig_data_format(cb_fusion);
                }
                cb_reserve_back(cb_im_or_out, in0_block_w);
                // DPRINT_PACK(DPRINT << "cb_reserve_back(cb_im_or_out, in0_block_w);" << " wt:" << wt <<" cb_im_or_out:"<<static_cast<uint32_t>(cb_im_or_out)<< ENDL());

                reconfig_data_format_srca(cb_fusion, cb_xmm);

                ACQ();
                mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    // cb_xmm[wt+wtr] since we pop in0_block_w * num_blocks_inner_dim from cb_xmm after the entire loop
                    // 第二次使用：应用归一化算子
                    mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, wt + wtr, 0, wtr);  // tile *= 1/(sum(exp(x)))
                    // DPRINT_MATH(DPRINT << "mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, wt + wtr, 0, wtr);" << " wt:" << wt << " wtr:" << wtr << ENDL());
                    pack_tile(wtr, cb_im_or_out);  // pack either to intermediate (cb_fusion or out0)
                }

                cb_push_back(cb_im_or_out, in0_block_w);  // if no gamma/beta are provided, this will be passed on to the writer
                // DPRINT_PACK(DPRINT << "cb_push_back(cb_im_or_out, in0_block_w);" << " wt:" << wt <<" cb_im_or_out:"<<static_cast<uint32_t>(cb_im_or_out)<< ENDL());
                REL();

                if constexpr (!(do_gamma == 0 )) {

                    reconfig_data_format_srca(cb_xmm, cb_fusion);

                }
                if constexpr (do_gamma) {
                    pack_reconfig_data_format(cb_norm_output);

                    reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
                    ACQ();
                    uint32_t cb_outg = cb_norm_output ;
                    mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                    cb_reserve_back(cb_outg, in0_block_w);
                    cb_wait_front(cb_gamma, wt + in0_block_w);  // we don't pop, TODO: only wait on first ht
                    cb_wait_front(cb_fusion, in0_block_w);
                    for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                        mul_tiles_bcast_rows(cb_fusion, cb_gamma, wtr, wt + wtr, wtr);  // tile *= 1/(sum(exp(x)))
                        pack_tile(wtr, cb_outg);  // pack either to intermediate (cb_fusion or out0)
                    }
                    cb_pop_front(cb_fusion, in0_block_w);
                    // we don't pop gamma
                    cb_push_back(cb_outg, in0_block_w);
                    // We don't pop gamma since it's 1,1,1,in0_block_w * num_blocks_inner_dim and we reuse it for all NCHt
                    REL();
                }
            }
            cb_pop_front(cb_ex2pe, 1);
            // DPRINT_UNPACK(DPRINT << "cb_pop_front(cb_ex2pe, 1);" << ", cb_ex2pe:"<<static_cast<uint32_t>(cb_ex2pe) << ENDL());
            cb_pop_front(cb_xmm, in0_block_w * num_blocks_inner_dim);   //因为size是Wt，所以对于norm的输入来说，肯定是全程都在L1（norm也需要两遍的使用）
            // DPRINT_UNPACK(
            //     DPRINT << "cb_pop_front(cb_xmm, in0_block_w * num_blocks_inner_dim);"
            //            << ", cb_xmm:" << static_cast<uint32_t>(cb_xmm)
            //            << ", in0_block_w * num_blocks_inner_dim:" << in0_block_w * num_blocks_inner_dim << ENDL());

            // 为了流水起来而cb_pop_front，实际上cb_norm_output应该是matmul的in0输入
            // for (uint32_t i = 0; i < in0_block_w * num_blocks_inner_dim; i += in0_block_w) {
            //     cb_wait_front(cb_norm_output, in0_block_w);
            //     cb_pop_front(cb_norm_output, in0_block_w);
            // }

//norm与matmul的分隔符-----------------------------------------------------------------------------------------------------------------------------

            mm_block_init(cb_im_or_out, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

            // 勿删：
            // 由于不知道对应layernorm.cpp中的cb_im_or_out哪个，所以直接与dst寄存器进行比较(dprint_tensix_dest_reg)
            // TileSlice的第二个参数代表第几个tile，如果是8、9、10、11，则对应着layernorm.cpp中的if (wt == 8)
            // {dprint_tensix_dest_reg(wtr);}，即打印wt从8开始的几个tile
            DPRINT_UNPACK({
                DPRINT << "=== cb_im_or_out ===, bh:" << bh << ENDL();
                DPRINT << TileSlice<128>(
                              cb_im_or_out,
                              8,
                              SliceRange{.h0 = 4, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 16, .ws = 1},
                              true,
                              false)
                       << ENDL();
                DPRINT << TileSlice<128>(
                              cb_im_or_out,
                              9,
                              SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              true,
                              false)
                       << ENDL();
                DPRINT << TileSlice<128>(
                              cb_im_or_out,
                              10,
                              SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              true,
                              false)
                       << ENDL();
                DPRINT << TileSlice<128>(
                              cb_im_or_out,
                              11,
                              SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                              true,
                              false)
                       << ENDL();

            })
            // DPRINT_UNPACK({
            //     DPRINT << "=== in1_cb_id ===, bh:" << bh << ENDL();
            //     DPRINT << TileSlice(
            //                   in1_cb_id,
            //                   0,
            //                   SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
            //                   true,
            //                   false)
            //            << ENDL();
            // })

            // 现在的输入是cb_norm_output，一整行已经在L1中。等待的是整个K(in0_block_w * num_blocks_inner_dim)
            cb_wait_front(cb_im_or_out, in0_block_w * num_blocks_inner_dim);
            uint32_t matmul_in0_index = 0;

            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {    // 1
                bool enable_reload = false;
                uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;



                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {   // num_blocks_inner_dim = num_blocks = K / in0_block_w = K(Kt)
                    bool last_out = block == (num_blocks_inner_dim - 1);

                    // cb_wait_front(in0_cb_id, in0_block_num_tiles);
                    // DPRINT_UNPACK(DPRINT << "cb_wait_front(in0_cb_id, in0_block_num_tiles);" << ENDL());
                    cb_wait_front(in1_cb_id, in1_block_num_tiles);
                    // DPRINT_UNPACK(DPRINT << "cb_wait_front(in1_cb_id, in1_block_num_tiles);" << ENDL());

                    // DPRINT_UNPACK({
                    //     DPRINT << "m=== cb_im_or_out ===, bh:" << bh <<" block: "<< block << ENDL();
                    //     DPRINT << TileSlice(
                    //                 cb_im_or_out, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws =
                    //                 1}, true, false)
                    //         << ENDL();
                    //     DPRINT << "matmul_in0_index: " << matmul_in0_index << ENDL();
                    // })

                    int in0_index_subblock_offset = 0;
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {// 1（与K无关）现在只有1行，因此固定为1，该循环可以去掉
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
                             in1_subblock++) {  // 8  输出矩阵在宽度维度的subblock数量
                            // DPRINT_UNPACK(DPRINT << "in1_subblock: " << in1_subblock << ", in1_num_subblocks: " <<
                            // in1_num_subblocks << ENDL());
                            tile_regs_acquire();
                            // DPRINT_MATH(DPRINT << "tile_regs_acquire();" <<" in1_subblock: "<<in1_subblock <<
                            // ENDL());
                            if (enable_reload) {
                                reload_from_cb_to_dst(
                                    // in0_cb_id,
                                    cb_im_or_out,
                                    in1_cb_id,
                                    mm_partials_cb_id,
                                    in1_transpose_tile,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

#ifndef SKIP_COMPUTE
                            // Compute output sub-block
                            uint32_t dst_index =
                                0;  // start at 0, each call to matmul_block internally increments dst_index
                            // in0_index和matmul_in0_index的转换有问题，现在过了前8次就出事。
                            uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                            uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                            matmul_in0_index = matmul_in0_index % 64;
                            // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                                // accumulation is done by iterating matmul_block across inner dim
                                // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride
                                // in0
                                // 对于in0只有1行的情况，matmul_block每一次调用计算的是[1*1*subblock_w]，随后在K维度叠加
                                matmul_block(
                                    // in0_cb_id,
                                    cb_im_or_out,
                                    in1_cb_id,
                                    // TODO:in0_index应该改变，因为cb_im_or_out相比于in0_cb_id处在循环的更外层
                                    // in0_index,   //指定矩阵A的当前列（K维度位置）
                                    matmul_in0_index,
                                    in1_index,  //指定矩阵B的当前行（K维度位置）
                                    dst_index,//subblock的意义就在于dst，dst_index就是一个输出subblock在dst寄存器的索引
                                    in1_transpose_tile,
                                    out_subblock_w, //matmul_block处理[out_subblock_h*out_subblock_w]的外积，因此in1_index不用照顾到out_subblock_w参数，matmul_block内部会处理out_subblock_h参数
                                    out_subblock_h, //matmul_block处理[out_subblock_h*out_subblock_w]的外积，因此in0_index不用照顾到out_subblock_h参数，matmul_block内部会处理out_subblock_w参数
                                    in0_block_w);
                                // DPRINT_MATH(DPRINT << "matmul_block();"<< "in1_subblock:" << in1_subblock <<"
                                // out_subblock_num_tiles:" << out_subblock_num_tiles << ENDL()); in0_index++; // stride
                                // right by 1
                                matmul_in0_index++;
                                in1_index += in1_block_w;  // to stride down by 1 need to stride by in_per_core_w
                                                           // (should be called in1_block_w)
                            }

#endif  // SKIP_COMPUTE

                            if (last_out) {

                                tile_regs_commit();
                                // Pack out to output buffer
                                cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                                // DPRINT_PACK(DPRINT << "cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);"<<
                                // "in1_subblock:" << in1_subblock << ENDL());
                                tile_regs_wait();

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC

                                PACK((llk_pack_reconfig_l1_acc(0)));

#endif

                                uint32_t start_dst_index = 0;
                                pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

                                // DPRINT_UNPACK({
                                //     DPRINT << "m=== mm_out_cb_id ===, bh:" << bh <<" block: "<< block << ENDL();
                                //     DPRINT << TileSlice(
                                //                 mm_out_cb_id, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1
                                //                 = 32, .ws = 1}, true, false)
                                //         << ENDL();
                                // })

                            } else {
                                tile_regs_commit();
                                // DPRINT_MATH(DPRINT << "tile_regs_commit();" << "in1_subblock:" << in1_subblock <<
                                // ENDL()); Wait for tiles in output buffer to be written out since interm and output
                                // share memory

                                if (block == 0) {
                                    cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                                    // DPRINT_PACK(DPRINT << "cb_reserve_back(out_cb_id, out_num_tiles_to_wait);" <<
                                    // "in1_subblock:" << in1_subblock << ENDL());
                                    out_num_tiles_to_wait += out_subblock_num_tiles;
                                }
                                // Move partial result to interm buffer
                                cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                                // DPRINT_PACK(DPRINT << "cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);"
                                // << "in1_subblock:" << in1_subblock << ENDL());
                                tile_regs_wait();
                                // DPRINT_PACK(DPRINT << "tile_regs_wait" << "in1_subblock:" << in1_subblock << ENDL());

#ifdef PACKER_L1_ACC
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                    DPRINT_PACK(DPRINT << "llk_pack_reconfig_l1_acc(0)" << "in1_subblock:" << in1_subblock << ENDL());
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                    DPRINT_PACK(DPRINT << "llk_pack_reconfig_l1_acc(1)" << "in1_subblock:" << in1_subblock << ENDL());
                                }
#endif

                                uint32_t start_dst_index = 0;
                                //将dst寄存器打包到CB
                                pack_tile_block(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);
                                // DPRINT_PACK(DPRINT << "pack_tile_block(start_dst_index, mm_partials_cb_id,
                                // out_subblock_num_tiles);" << "in1_subblock:" << in1_subblock<<"
                                // out_subblock_num_tiles:"<<out_subblock_num_tiles << ENDL());

                                tile_regs_release();
                                // DPRINT_PACK(DPRINT << "tile_regs_release" << "in1_subblock:" << in1_subblock <<
                                // ENDL());
                                cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                                // DPRINT_PACK(DPRINT << "cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);" <<
                                // "in1_subblock:" << in1_subblock << ENDL());

                                // DPRINT_UNPACK({
                                //     DPRINT << "m=== mm_partials_cb_id ===, bh:" << bh <<" block: "<< block << ENDL();
                                //     DPRINT << TileSlice(
                                //                 mm_partials_cb_id, 0, SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0,
                                //                 .w1 = 32, .ws = 1}, true, false)
                                //         << ENDL();
                                // })
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }  // in1_num_subblocks
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }   //in0_num_subblocks

#ifdef PACKER_L1_ACC

                    // Last iteration does spill and reload to output buffer
                    if (block < num_blocks_inner_dim - 2) {
                        cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
                        cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }  // reload when last iteration

#else
                    if constexpr (spill) {
                        enable_reload = true;
                    }
#endif

                    // 这个时候还不能pop
                    // cb_pop_front(in0_cb_id, in0_block_num_tiles);
                    // DPRINT_PACK(DPRINT << "cb_pop_front(in0_cb_id, in0_block_num_tiles);" << ENDL());
                    cb_pop_front(in1_cb_id, in1_block_num_tiles);
                    // DPRINT_PACK(DPRINT << "cb_pop_front(in1_cb_id, in1_block_num_tiles);" << ENDL());
                }   //num_blocks_inner_dim


                // 与PACK((pack_reconfig_data_format(mm_partials_cb_id)));对应，将srca从mm_partials_cb_id切换为in1_cb_id
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {

                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
                    // reconfigure init for matmul
                    mm_block_init_short(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
            // 这一行结束计算了，所以应该pop
            cb_pop_front(cb_im_or_out, in0_block_w * num_blocks_inner_dim);
        }
        // DPRINT << "for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) END END END" << ENDL();
    }
    // DPRINT << "for (uint32_t b = 0; b < batch; b++) END END END" << ENDL();
}
}  // namespace NAMESPACE
