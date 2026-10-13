#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp8_detail {

// V4 tile 类型集（TileN=64：宽输入 tile + 32 列切片 + 窄链 + 输出）
template <typename InT, int TileMv, int ValidRows, int WIDE, int BS>
struct TileTypes {
    using t_blk = VecTileM32<InT,        TileMv, WIDE, ValidRows, WIDE>;
    using t_cut = VecTileM32<InT,        TileMv, BS,   ValidRows, BS>;
    using t_rw  = VecTileM32<InT,        TileMv, BS,   ValidRows, 1>;
    using t_rin = VecTileM32<InT,        TileMv, 1,    ValidRows, 1>;
    using t_row = VecTileM32<__bf16,     TileMv, 1,    ValidRows, 1>;
    using t_frw = VecTileM32<float,      TileMv, 1,    ValidRows, 1>;
    using t_e8b = VecTileM32<__fp8_e8m0, TileMv, 1,    ValidRows, 1>;
    using t_o   = VecTileM32<__fp8_e4m3, TileMv, BS,   ValidRows, BS>;
};

// 单个 32 列 MX 块的完整量化链（归约→floor→scale→recip 守卫→直乘→y 窄存）。
//   normalization 模式：命名函数 + always_inline + 裸指针/标量参数（无闭包），
//   SubTileView 在函数体内从 BorrowedTileArray 创建。
//   参数：absfrags/infrags = TPARTVIEW 结果（parent absx/xin 的切片数组）；
//         zero_in = 共享零 tile（TADD materialize 用，跨块复用）；
//         blk = 块编号（0..WIDE/BS-1，运行期索引）。
template <typename InT, typename TT, typename BorrowedArr,
          int BlockSize, int TileMv, int M, int N, int scaleCols, uint16_t RECIP_EMAX>
__attribute__((always_inline))
inline void quantize_block(
    BorrowedArr &absfrags, BorrowedArr &infrags,
    typename TT::t_rin &zero_in,
    __fp8_e4m3 *y, uint8_t *scale,
    int64_t row0, int64_t kb_col, int blk) {
    using namespace pto;
    using t_rw  = typename TT::t_rw;
    using t_rin = typename TT::t_rin;
    using t_row = typename TT::t_row;
    using t_frw = typename TT::t_frw;
    using t_e8b = typename TT::t_e8b;
    using t_cut = typename TT::t_cut;
    using t_o   = typename TT::t_o;
    using gm_y  = global_tensor<__fp8_e4m3, RowMajor<M, N>>;
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    auto absfrag = absfrags[0][blk];
    auto infrag  = infrags[0][blk];

    t_rw  mw; TROWMAX(mw, absfrag);
    auto  mv = TREDUCEPREFIXVIEW<t_rin>(mw);
    t_rin mi; TADD(mi, zero_in, mv);

    t_row sh, ni, nz, ns, k, rp;
    auto shu = reinterpret_tile<uint16_t>(sh);
    auto niu = reinterpret_tile<uint16_t>(ni);
    auto nzu = reinterpret_tile<uint16_t>(nz);
    auto nsu = reinterpret_tile<uint16_t>(ns);
    auto ku  = reinterpret_tile<uint16_t>(k);
    auto rpu = reinterpret_tile<uint16_t>(rp);
    if constexpr (std::is_same_v<InT, __bf16>) {
        auto mu = reinterpret_tile<uint16_t>(mi);
        TANDS(mu, mu, BF16_EXP_MASK);
        TCMPS<CmpMode::NE>(niu, mu, BF16_EXP_MASK);
        TCMPS<CmpMode::NE>(nzu, mu, static_cast<uint16_t>(0));
        TMULS(sh, mi, __builtin_bit_cast(__bf16, RECIP_EMAX));
    } else {
        t_frw mf; TCVT(mf, mi);
        auto mu = reinterpret_tile<uint32_t>(mf);
        TANDS(mu, mu, FP32_EXP_MASK);
        t_row mb; TCVT(mb, mf);
        auto mbu = reinterpret_tile<uint16_t>(mb);
        TCMPS<CmpMode::NE>(niu, mbu, BF16_EXP_MASK);
        TCMPS<CmpMode::NE>(nzu, mbu, static_cast<uint16_t>(0));
        TMULS(sh, mb, __builtin_bit_cast(__bf16, RECIP_EMAX));
    }
    t_e8b sb; TCVT(sb, sh);
    global_iterator<gm_s, t_e8b> si(
        reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb_col + blk);
    auto gsi = si(0, 0); TSTORE(gsi, sb);

    TEXPANDS(ku, BF16_EXP_BIAS);
    TSUB(rpu, ku, shu);
    TCMPS<CmpMode::NE>(nsu, shu, BF16_EXP_BIAS);
    TSELS(rpu, niu, BF16_NAN_PATTERN, rpu);
    TSELS(rpu, nzu, static_cast<uint16_t>(0), rpu);
    TSELS(rpu, nsu, BF16_SPECIAL_EXP, rpu);

    t_cut norm;
    if constexpr (std::is_same_v<InT, __bf16>) {
        TROWEXPANDMUL(norm, infrag, rp);
    } else {
        t_frw rf; TCVT(rf, rp);
        TCVT(norm, infrag);
        TROWEXPANDMUL(norm, norm, rf);
    }
    t_o oq;
    TCVT(oq, norm);
    global_iterator<gm_y, t_o> yb(y + row0 * N + (kb_col + blk) * BlockSize);
    auto gyb = yb(0, 0); TSTORE(gyb, oq);
}

} // namespace tail_ocp_fp8_detail

// ===========================================================================
// TAIL-OCP-FP8 正式 kernel（V4：TileN=64 + TSELS 守卫 + bf16 直乘 + 2D 切分）
//   固定 SPMD 4-PE。InT(bf16/half/fp32) in / e4m3 out / e8m0 scale / BS=32。
//   每 (row-tile, 64-col-tile)：1 次宽 TLOAD(128B/行) + 1 次整体 TABS +
//   TPARTVIEW 切 2 片 → 2 × quantize_block（命名函数，无 lambda）。
// ===========================================================================
template <int M, int N, int BlockSize = 32, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8(InT *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(N % 64 == 0, "V4 TileN=64: N must be multiple of 64");
    static_assert(BlockSize % 32 == 0, "BlockSize must be multiple of 32");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>();
    constexpr int numKb     = N / BlockSize;
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr bool oddTail  = (numKb % 2) != 0;
    constexpr int kPeNum    = 4;
    constexpr int WIDE      = 64;
    constexpr int BlocksPerTile = WIDE / BlockSize;  // 2

    const uint32_t tid = get_thread_idx();

    using gm_x = global_tensor<InT, RowMajor<M, N>>;

    auto process_tile = [&]<int TileMv, int ValidRows>(int row0, int kbBegin,
                                                       int kbCount, bool ownsLastKb) {
        using TT = tail_ocp_fp8_detail::TileTypes<InT, TileMv, ValidRows, WIDE, BlockSize>;
        using t_blk = typename TT::t_blk;
        using t_cut = typename TT::t_cut;
        using t_rin = typename TT::t_rin;
        using t_e8b = typename TT::t_e8b;
        using t_row = typename TT::t_row;
        using t_frw = typename TT::t_frw;

        for (int kb = kbBegin; kb < kbBegin + kbCount; ++kb) {
            const int kb0 = kb * BlocksPerTile;
            global_iterator<gm_x, t_blk> x_iter(x + row0 * N + kb0 * BlockSize);
            auto gx = x_iter(0, 0);
            t_blk xin; TLOAD(xin, gx);
            t_blk absx; TABS(absx, xin);
            auto absfrags = TPARTVIEW<t_cut, 1, BlocksPerTile>(absx);
            auto infrags  = TPARTVIEW<t_cut, 1, BlocksPerTile>(xin);

            t_rin zero_in; TEXPANDS(zero_in, static_cast<InT>(0.0f));

            for (int blk = 0; blk < BlocksPerTile; ++blk) {
                tail_ocp_fp8_detail::quantize_block<
                    InT, TT,
                    decltype(absfrags),
                    BlockSize, TileMv, M, N, scaleCols, RECIP_EMAX>(
                    absfrags, infrags, zero_in, y, scale, row0, kb0, blk);
            }
        }

        if constexpr (oddTail) {
            if (ownsLastKb) {
                t_row zpad_bf;
                TEXPANDS(zpad_bf, __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x0040)));
                t_e8b zpad;
                TCVT(zpad, zpad_bf);
                global_iterator<global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>, t_e8b> zs_iter(
                    reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + numKb);
                auto gzs = zs_iter(0, 0); TSTORE(gzs, zpad);
            }
        }
    };

    auto run_pe = [&]<int Pe>() {
        constexpr int TileM = 32;
        constexpr int rowTiles = (M + TileM - 1) / TileM;
        constexpr int peRows   = rowTiles < kPeNum ? rowTiles : kPeNum;
        constexpr int peCols   = kPeNum / peRows;
        constexpr int rowLane  = Pe / peCols;
        constexpr int colLane  = Pe % peCols;
        if constexpr (rowLane < peRows) {
            constexpr int tilesBase = rowTiles / peRows;
            constexpr int tilesRem  = rowTiles % peRows;
            constexpr int myTiles   = tilesBase + (rowLane < tilesRem ? 1 : 0);
            constexpr int myTileBegin =
                rowLane * tilesBase + (rowLane < tilesRem ? rowLane : tilesRem);
            constexpr int numTiles = numKb / BlocksPerTile;
            constexpr int kbBase  = numTiles / peCols;
            constexpr int kbRem   = numTiles % peCols;
            constexpr int myKb    = kbBase + (colLane < kbRem ? 1 : 0);
            constexpr int kbBegin = colLane * kbBase + (colLane < kbRem ? colLane : kbRem);
            constexpr bool ownsLastKb = (kbBegin + myKb == numTiles);
            if constexpr (myTiles > 0 && myKb > 0) {
                constexpr bool ownLastTile = (myTileBegin + myTiles == rowTiles);
                constexpr int lastValid = (M % TileM) != 0 ? (M % TileM) : TileM;
                constexpr int myFull = ownLastTile ? myTiles - 1 : myTiles;
                for (int t = 0; t < myFull; ++t) {
                    process_tile.template operator()<TileM, TileM>(
                        (myTileBegin + t) * TileM, kbBegin, myKb, ownsLastKb);
                }
                if constexpr (ownLastTile) {
                    process_tile.template operator()<TileM, lastValid>(
                        (rowTiles - 1) * TileM, kbBegin, myKb, ownsLastKb);
                }
            }
        }
    };

    switch (static_cast<int>(tid)) {
        case 0: run_pe.template operator()<0>(); break;
        case 1: run_pe.template operator()<1>(); break;
        case 2: run_pe.template operator()<2>(); break;
        case 3: run_pe.template operator()<3>(); break;
        default: break;
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
