// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
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

#include "tt_metal/hw/inc/debug/dprint_tensix_pack.h"
#include "debug/dprint_pages.h"


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
    //constexpr auto cb_out = tt::CBIndex::c_4;    // output
    constexpr auto cb_gamma = tt::CBIndex::c_10;
    constexpr uint32_t cb_xmm = in0_cb_id;  // x minus mean
    constexpr auto cb_ex2 = tt::CBIndex::c_19;     // 存储 E[x²],即 x 平方的均值
    constexpr auto cb_xmm2 = tt::CBIndex::c_20;    // 存储 x²,即每个元素的平方           
    constexpr auto cb_ex2pe = tt::CBIndex::c_21;   // 存储 1/√(E[x²] + ε),即 RMS 归一化因子
    constexpr auto cb_fusion = tt::CBIndex::c_22;  // stream gamma/beta
    constexpr auto scaler0 = 0;
    constexpr auto cb_norm_output = tt::CBIndex::c_16;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    constexpr int cb_im_or_out = (do_gamma ) ? cb_fusion : cb_norm_output;

    //constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;
    constexpr uint32_t mm_out_cb_id = out_cb_id;



#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;
    //constexpr bool spill = false;//test

    // DPRINT << "Compute kernel started" << ENDL();
    // DPRINT << "compute num_blocks_w_dim"<<num_blocks_w_dim<<ENDL();

    cb_wait_front(cb_scaler, 1);  // comes from the reader
            DPRINT_UNPACK({  
                DPRINT << "=== cb_scaler ===" << ENDL();  
                DPRINT << TileSlice(cb_scaler, 0,   
                    SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},   
                    true, false) << ENDL();     
            })



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

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {// 8

            /*
                rmsnorm的计算
            */
            constexpr int onetile = 1;
            constexpr int dst0 = 0;

            // X+Y
            reconfig_data_format(cb_xmm, cb_xmm);
            pack_reconfig_data_format(cb_xmm2);

            // 第一步：计算 x²
            // cb_xmm * cb_xmm => cb_xmm2
            mul_tiles_init(cb_xmm, cb_xmm);
            for (uint32_t wt = 0; wt < in0_block_w * num_blocks_inner_dim; wt += in0_block_w) {     // num_blocks_inner_dim = num_blocks = K / in0_block_w = K(Kt)
                DPRINT_UNPACK(DPRINT << "compute start cb_wait_front(cb_xmm, wt + in0_block_w);"<<", cb_xmm:"<<static_cast<uint32_t>(cb_xmm)<<", wt:"<<wt<<", in0_block_w:"<<in0_block_w <<", bh"<<bh<< ENDL());
                cb_wait_front(cb_xmm, wt + in0_block_w);  // cumulative wait
                DPRINT_UNPACK({ 
                    DPRINT << "=== cb_xmm ===, bh:"<<bh <<", wt:"<<wt<< ENDL();  
                    DPRINT << TileSlice(cb_xmm, 0,  
                        SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},  
                        true, false) << ENDL();     
                })
                cb_reserve_back(cb_xmm2, in0_block_w);  // can probably use less space for this if we block
                // ACQ();
                tile_regs_acquire();
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    // 第一次使用：计算x^2
                    mul_tiles(cb_xmm, cb_xmm, wt + wtr, wt + wtr, wtr);
                    if (bh==0){
                        DPRINT_MATH({ DPRINT << "xmm2 dst_index:" << wtr << ENDL(); })
                        dprint_tensix_dest_reg(wtr);
                    }

                    //pack_tile(wtr, cb_xmm2);
                }
                tile_regs_commit();  
                tile_regs_wait(); 
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    pack_tile(wtr, cb_xmm2);
                }
                tile_regs_release();
                cb_push_back(cb_xmm2, in0_block_w);
                //REL();
            }
            
            reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
                       
            // 第二步：计算E[x^2]归一化因子
            // 这个循环对 cb_xmm2 中的所有平方值进行归约求和,然后乘以 cb_scaler 得到均值,结果存入 cb_ex2。
            if constexpr (FLOAT32_DTYPE) {
                reconfig_data_format(cb_xmm2, cb_scaler);
            }
            cb_reserve_back(cb_ex2, 1);  // 在输出缓冲区 cb_ex2 中预留 1 个 tile 的空间用于存储结果
            reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, cb_ex2);
            //ACQ();
            cb_wait_front(cb_xmm2, in0_block_w * num_blocks_inner_dim); //累积等待整行的所有 in0_block_w * num_blocks_inner_dim 个 tile 都准备好 这确保了归约操作可以访问完整的一行数据
            DPRINT_UNPACK({ 
                DPRINT << "=== cb_xmm2 ===, bh:"<<bh << ENDL();  
                DPRINT << TileSlice(cb_xmm2, 0,   
                    SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},   
                    true, false) << ENDL();     
            })

            tile_regs_acquire(); 
            // 外层循环以 in0_block_w 为步长遍历所有 tile
            for (uint32_t wt = 0; wt < in0_block_w * num_blocks_inner_dim; wt += in0_block_w) {
                // reduce
                // 内层循环对每个 tile 调用 reduce_tile,将其累加到目标寄存器 dst0 中
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, wt + wtr, scaler0, dst0);
                    // DPRINT_MATH(DPRINT << "reduce_tile, cb_xmm2:"<<static_cast<uint32_t>(cb_xmm2)<<", wt:"<<wt<<", wtr:"<<wtr<<", scaler0:"<<scaler0<<", dst0:"<<dst0 << ENDL());
                }
            }
            cb_pop_front(cb_xmm2, in0_block_w * num_blocks_inner_dim);
            tile_regs_commit();  
            tile_regs_wait();  
            pack_tile(dst0, cb_ex2);
            reduce_uninit();
            //REL();
            tile_regs_release(); 

            cb_push_back(cb_ex2, 1);
            cb_wait_front(cb_ex2, 1);
            DPRINT_UNPACK({  
                DPRINT << "=== cb_ex2 ===, bh:"<<bh << ", Wt: "<<in0_block_w * num_blocks_inner_dim << ENDL();  
                DPRINT << TileSlice(cb_ex2, 0,   
                    SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},   
                    true, false) << ENDL();     
            })

            /* Var(x) + eps
            * add epsilon E[(x-E[x])^2]+eps
            */
            if constexpr (FLOAT32_DTYPE) {
                reconfig_data_format(cb_ex2, cb_eps);
            }
            // ACQ();
            tile_regs_acquire();  
            add_tiles_init(cb_ex2, cb_eps);
            add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

            rsqrt_tile_init<LEGACY_RSQRT>();
            rsqrt_tile<LEGACY_RSQRT>(dst0);

            tile_regs_commit();  
            cb_reserve_back(cb_ex2pe, 1);  // 1
            tile_regs_wait(); 
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();  
            cb_push_back(cb_ex2pe, 1);
            // REL();
            cb_pop_front(cb_ex2, 1);

            /* ln(x) * gamma + beta (gamma and beta are optional)
            * now xmm = (x-E[x])
            * we have 1.0/sqrt( E[(x-E[x])^2] + eps) in cb_ex2pe
            * just need to bcast_mul xmm with cb_ex2pe
            */
            // 第三步：分块归一化循环，这个循环将 cb_xmm 与归一化因子 cb_ex2pe (即 1/√(E[x²]+ε)) 相乘，结果存入
            // cb_fusion
            cb_wait_front(cb_ex2pe, 1);
            DPRINT_UNPACK({  
                DPRINT << "=== cb_ex2pe ===, bh:"<<bh << ENDL();  
                DPRINT << TileSlice(cb_ex2pe, 0,   
                    SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},   
                    true, false) << ENDL();     
            })
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
                DPRINT_PACK(DPRINT << "compute cb_reserve_back(cb_im_or_out, in0_block_w);" << " wt:" << wt <<" cb_im_or_out:"<<static_cast<uint32_t>(cb_im_or_out)<<",bh:"<<bh << ENDL());

                reconfig_data_format_srca(cb_fusion, cb_xmm);

                ACQ();
                mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
                for (uint32_t wtr = 0; wtr < in0_block_w; wtr++) {
                    // cb_xmm[wt+wtr] since we pop in0_block_w * num_blocks_inner_dim from cb_xmm after the entire loop
                    // 第二次使用：应用归一化算子
                    mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, wt + wtr, 0, wtr);  // tile *= 1/(sum(exp(x)))
                    pack_tile(wtr, cb_im_or_out);  // pack either to intermediate (cb_fusion or out0)
                }
                
                cb_push_back(cb_im_or_out, in0_block_w);  // if no gamma/beta are provided, this will be passed on to the writer
                DPRINT_PACK(DPRINT << "compute cb_push_back(cb_im_or_out, in0_block_w);" << " wt:" << wt <<" cb_im_or_out:"<<static_cast<uint32_t>(cb_im_or_out)<<",bh:"<<bh << ENDL());
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
            DPRINT_UNPACK(DPRINT << "compute cb_pop_front(cb_ex2pe, 1);" << ", cb_ex2pe:"<<static_cast<uint32_t>(cb_ex2pe) <<",bh:"<<bh << ENDL());
            cb_pop_front(cb_xmm, in0_block_w * num_blocks_inner_dim);
            DPRINT_UNPACK(DPRINT << "compute cb_pop_front(cb_xmm, in0_block_w * num_blocks_inner_dim);" << ", cb_xmm:"<<static_cast<uint32_t>(cb_xmm)
                <<", in0_block_w * num_blocks_inner_dim:"<<in0_block_w * num_blocks_inner_dim <<",bh:"<<bh << ENDL());
//norm与matmul的分隔符-----------------------------------------------------------------------------------------------------------------------------
            // if (bh == 0) {
            // reconfig_data_format_srca(cb_im_or_out);
            // reconfig_data_format_srcb(in1_cb_id);
            pack_reconfig_data_format(mm_out_cb_id);
            mm_block_init_short(cb_im_or_out, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
            // mm_block_init(cb_im_or_out, in1_cb_id, mm_out_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
            // } 
            cb_wait_front(cb_im_or_out, in0_block_num_tiles * num_blocks_inner_dim);//K
            DPRINT_UNPACK(DPRINT << "compute cb_wait_front(cb_im_or_out, in0_block_num_tiles);in0_block_num_tiles:"<<in0_block_num_tiles<<", bh:"<<bh<< ENDL());
            DPRINT_UNPACK({  
                DPRINT << "=== Input Matrix A ===" << ENDL();  
                DPRINT << TileSlice(cb_im_or_out, 0,   
                    SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},   
                    true, false) << ENDL();     
            })
            //可以删除的日志
            // for (uint8_t r = 0; r < 32; ++r) {  
            //     SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};  
            //     DPRINT_UNPACK({   
            //         DPRINT << "Row " << r << ": "   
            //             << TileSlice(cb_im_or_out, 0, sr, true, false)   
            //             << ENDL();   
            //     });  
            // }  

            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {    // per_core_N / out_block_w = 8
                // if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                if ( bw > 0) {
                    // reconfig_data_format_srcb(in1_cb_id);
                    pack_reconfig_data_format(mm_out_cb_id);
                    //mm_block_init(cb_im_or_out, in1_cb_id, mm_out_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                    mm_block_init_short(cb_im_or_out, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
                // }
                bool enable_reload = false;

                int in0_index_subblock_offset = 0;
                DPRINT_MATH(DPRINT << "start tile_regs_acquire"<<", bh:"<<bh <<", bw:"<<bw << ENDL());
                tile_regs_acquire();
                DPRINT_MATH(DPRINT << "tile_regs_acquire"<<", bh:"<<bh <<", bw:"<<bw << ENDL());
                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {   // K / in0_block_w = 16 / 8 = 2
                    bool last_out = block == (num_blocks_inner_dim - 1);

                    cb_wait_front(in1_cb_id, in1_block_num_tiles);//8
                    DPRINT_UNPACK(DPRINT << "compute cb_wait_front in1_block_num_tiles: "<<in1_block_num_tiles <<", bh:"<<bh <<", bw:"<<bw 
                        << ", block:"<<block << ENDL());
                    DPRINT_UNPACK({                          
                        DPRINT << "=== Matrix B ===" <<", bw:"<<bw <<", bh:"<<bh << ENDL();  
                        DPRINT << TileSlice(in1_cb_id, 0,  
                            SliceRange{.h0=0,.h1=32,.hs=1,.w0=0,.w1=32,.ws=1},  
                            true, false) << ENDL();  
                    })  
                    // for (uint8_t r = 0; r < 32; ++r) {  
                    //     SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};  
                    //     DPRINT_UNPACK({   
                    //         DPRINT << "Row " << r << ": "   
                    //             << TileSlice(in1_cb_id, 0, sr, true, false)   
                    //             << ENDL();   
                    //     });  
                    // }  

#ifndef SKIP_COMPUTE
                    // Compute output sub-block
                    uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                    uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                    uint32_t in1_index = 0;  // offset into in1 block

                    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                        DPRINT_MATH(DPRINT << "compute start matmul_block()" <<", in0_index:"<<in0_index<<", in1_index:"<<in1_index
                            <<", dst_index:"<<dst_index<<", out_subblock_w: "<<out_subblock_w<<", out_subblock_h: "<<out_subblock_h
                            <<", in0_block_w: "<<in0_block_w<<", bh:"<<bh <<", bw:"<<bw << ", block:"<<block<< ENDL());
                        matmul_block(
                            cb_im_or_out,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            in1_transpose_tile,
                            out_subblock_w,
                            out_subblock_h,
                            in0_block_w);
                        DPRINT_MATH(DPRINT << "compute matmul_block()"<< ", in0_index:"<<in0_index<<", in1_index:"<<in1_index<<", dst_index:"<<dst_index
                            <<", bh:"<<bh <<", bw:"<<bw << ", block:"<<block<< ENDL());
                        DPRINT_MATH({ DPRINT << "----------------dst_index:" << dst_index << ENDL(); })
                        dprint_tensix_dest_reg(dst_index);
                        in0_index++; 
                        // in1_index += in1_block_w;//only for test
                        in1_index++;//only for test
                    }
#endif  // SKIP_COMPUTE

                    if (last_out) {
                        DPRINT_PACK(DPRINT << "compute start last cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);"<<", bw:"<<bw << ", block:"<<block
                            <<", bh:"<<bh << ENDL());
                        tile_regs_commit();
                        // Pack out to output buffer
                        cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
                        DPRINT_PACK(DPRINT << "compute last cb_reserve_back mm_out_cb_id, out_subblock_num_tiles: "<<out_subblock_num_tiles<<", bw:"<<bw 
                            <<", block:"<<block<<", bh:"<<bh<< ENDL());
                        tile_regs_wait();

// #if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
//                         PACK((pack_reconfig_data_format(mm_out_cb_id)));
// #endif

#ifdef PACKER_L1_ACC

                                PACK((llk_pack_reconfig_l1_acc(0)));

#endif

                        uint32_t start_dst_index = 0;
                        pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
                        tile_regs_release();
                        DPRINT_PACK(DPRINT << "tile_regs_release"<<", bh:"<<bh <<", bw:"<<bw <<", block:"<<block<< ENDL());
                        cb_push_back(mm_out_cb_id, out_subblock_num_tiles);
                        // std::array<ckernel::packer::pack_config_t, ckernel::packer::NUM_PACKERS> pack_config_vec;  
                        // MATH(pack_config_vec = ckernel::packer::read_pack_config();)  
                        // dprint_tensix_pack_config(); 
                    }
                    in0_index_subblock_offset += in0_subblock_num_tiles;

                    cb_pop_front(in1_cb_id, in1_block_num_tiles);
                    DPRINT_UNPACK(DPRINT << "compute cb_pop_front(in1_cb_id, in1_block_num_tiles);" <<", bw:"<<bw << ", block:"<<block<<", bh:"<<bh << ENDL());
                }   //num_blocks_inner_dim
            }
            cb_pop_front(cb_im_or_out, in0_block_num_tiles * num_blocks_inner_dim);
            DPRINT_UNPACK(DPRINT << "compute cb_pop_front(cb_im_or_out, in0_block_num_tiles);" <<", bh:"<<bh << ENDL());
        }
    }
    DPRINT_PACK(DPRINT << "compute bmm_large_block_zm_fused_bias_activation_fuse_norm complete"<< ENDL());
}
}  // namespace NAMESPACE