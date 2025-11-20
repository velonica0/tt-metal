// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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

// rmsnorm的compute kernel
namespace NAMESPACE {
void MAIN {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);                //宽度维度的 tile 数量
    constexpr uint32_t blk = get_compile_time_arg_val(1);               //core单次处理多少个 tile
    constexpr uint32_t do_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t do_beta = get_compile_time_arg_val(3);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(4) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(5) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // TODO(AP): check that if DST is indeed zeroed by release_dst (and initially), we can use it as zeroes

    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_scaler = tt::CBIndex::c_2;  // single tile 存储缩放因子 1/W,用于计算均值 它用于计算 E[x²],即对所有 x² 值求和后乘以 1/W
    constexpr auto cb_eps = tt::CBIndex::c_3;     // single tile 存储 epsilon 值,用于数值稳定性
    constexpr auto cb_in = tt::CBIndex::c_0;      // input x or a for fused pre-add (x=a+b)
    // constexpr auto cb_inb = tt::CBIndex::c_1;     // input b for fused pre-add
    constexpr auto cb_out = tt::CBIndex::c_16;    // output
    constexpr auto cb_gamma = tt::CBIndex::c_5;
    // constexpr auto cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_xmm = cb_in;  // x minus mean
    // constexpr auto cb_ex = tt::CBIndex::c_18;      // E[x]
    constexpr auto cb_ex2 = tt::CBIndex::c_19;     // 存储 E[x²],即 x 平方的均值
    constexpr auto cb_xmm2 = tt::CBIndex::c_20;    // 存储 x²,即每个元素的平方           
    constexpr auto cb_ex2pe = tt::CBIndex::c_21;   // 存储 1/√(E[x²] + ε),即 RMS 归一化因子
    constexpr auto cb_fusion = tt::CBIndex::c_22;  // stream gamma/beta
    constexpr auto scaler0 = 0;

    // constexpr uint32_t cb_x = cb_in;


    binary_op_init_common(cb_in, cb_in, cb_xmm2);


    cb_wait_front(cb_scaler, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader

    constexpr int cb_im_or_out = (do_gamma | do_beta) ? cb_fusion : cb_out;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

/*
 * X + Y
 */

        reconfig_data_format(cb_in, cb_in);
        pack_reconfig_data_format(cb_xmm2);



        // 第一步：计算 x²
        // 结果存入 cb_xmm2
        mul_tiles_init(cb_xmm, cb_xmm);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_xmm, wt + blk);  // cumulative wait
            cb_reserve_back(cb_xmm2, blk);    // can probably use less space for this if we block
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                //第一次使用：计算x^2
                mul_tiles(cb_xmm, cb_xmm, wt + wtr, wt + wtr, wtr);     
                // mul_tiles(cb_xmm, cb_col1, wt+wtr, wt+wtr, wtr);
                pack_tile(wtr, cb_xmm2);
            }
            cb_push_back(cb_xmm2, blk);
            REL();
        }

        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);


        // 第二步：计算E[x^2]归一化因子
        // 这个循环对 cb_xmm2 中的所有平方值进行归约求和,然后乘以 cb_scaler 得到均值,结果存入 cb_ex2。
        if constexpr (FLOAT32_DTYPE) {
            reconfig_data_format(cb_xmm2, cb_scaler);
        }
        cb_reserve_back(cb_ex2, 1); //在输出缓冲区 cb_ex2 中预留 1 个 tile 的空间用于存储结果
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, cb_ex2);
        ACQ();
        cb_wait_front(cb_xmm2, Wt); //累积等待整行的所有 Wt 个 tile 都准备好 这确保了归约操作可以访问完整的一行数据
        // cb_wait_front(cb_xmm, Wt);

        // 外层循环以 blk 为步长遍历所有 tile
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            // reduce
            // 内层循环对每个 tile 调用 reduce_tile,将其累加到目标寄存器 dst0 中
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, wt + wtr, scaler0, dst0);
            }
        }
        cb_pop_front(cb_xmm2, Wt);
        pack_tile(dst0, cb_ex2);
        reduce_uninit();
        REL();

        cb_push_back(cb_ex2, 1);
        cb_wait_front(cb_ex2, 1);

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

        /* ln(x) * gamma + beta (gamma and beta are optional)
         * now xmm = (x-E[x])
         * we have 1.0/sqrt( E[(x-E[x])^2] + eps) in cb_ex2pe
         * just need to bcast_mul xmm with cb_ex2pe
         */
        // 第三步：分块归一化循环，这个循环将 cb_xmm 与归一化因子 cb_ex2pe (即 1/√(E[x²]+ε)) 相乘，结果存入 cb_fusion
        cb_wait_front(cb_ex2pe, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            // if (ht == 1) UNPACK(( DPRINT << "wt_2=" << wt << " " ));
            // if (ht == 1) UNPACK(( DPRINT << "rem_2=" << rem << ENDL() ));
            reconfig_data_format(cb_xmm, cb_ex2pe);
            if constexpr (do_gamma == 0 && do_beta == 0) {
                pack_reconfig_data_format(cb_out);
            } else {
                pack_reconfig_data_format(cb_fusion);
            }
            cb_reserve_back(cb_im_or_out, blk);

            reconfig_data_format_srca(cb_fusion, cb_xmm);

            ACQ();
            mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                // cb_xmm[wt+wtr] since we pop Wt from cb_xmm after the entire loop
                // 第二次使用：应用归一化算子
                mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, wt + wtr, 0, wtr);  // tile *= 1/(sum(exp(x)))
                pack_tile(wtr, cb_im_or_out);  // pack either to intermediate (cb_fusion or out0)
            }
            cb_push_back(cb_im_or_out, blk);  // if no gamma/beta are provided, this will be passed on to the writer
            REL();

            if constexpr (!(do_gamma == 0 && do_beta == 0)) {

                reconfig_data_format_srca(cb_xmm, cb_fusion);

            }
            if constexpr (do_gamma) {
                if constexpr (do_beta == 0) {
                    pack_reconfig_data_format(cb_out);
                }
                reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
                ACQ();
                uint32_t cb_outg = do_beta ? cb_fusion : cb_out;
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                cb_reserve_back(cb_outg, blk);
                cb_wait_front(cb_gamma, wt + blk);  // we don't pop, TODO: only wait on first ht
                cb_wait_front(cb_fusion, blk);
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, wtr, wt + wtr, wtr);  // tile *= 1/(sum(exp(x)))
                    pack_tile(wtr, cb_outg);  // pack either to intermediate (cb_fusion or out0)
                }
                cb_pop_front(cb_fusion, blk);
                // we don't pop gamma
                cb_push_back(cb_outg, blk);
                // We don't pop gamma since it's 1,1,1,Wt and we reuse it for all NCHt
                REL();
            }
        }
        cb_pop_front(cb_ex2pe, 1);
        cb_pop_front(cb_xmm, Wt);   //因为size是Wt，所以对于norm的输入来说，肯定是全程都在L1（norm也需要两遍的使用）

    }  // NCHt loop
    // cb_pop_front(cb_scaler, 1); // optional for correctness
    // cb_pop_front(cb_eps, 1); // optional for correctness
    // cb_pop_front(cb_col1, 1); // optional for correctness
}
}  // namespace NAMESPACE
