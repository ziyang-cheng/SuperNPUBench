#ifndef SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
#define SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

// ============================================================================
// dynamic_hi_f4_quant —— 尾轴 encode,V1 蓝本 + ROW_MAJOR
// ============================================================================
// 权威格式/算法见 DESIGN.md §1/§2.1 与文档包 HIF4_QUANT_BF16.md;缺口见 RECORD.md。
//
// 2026-09-18 布局定案 = ROW_MAJOR(全程 Vec/RowMajor):
//   - 归约:`[32,4] → [32,1]` 行归约,输出 col=1,下游 `[32,1]` 消费 physCol=1 天然匹配
//     (mxquant 同款,过 gfrun)。SSM #685 只在归约 dst 声明成物理 `[32,2]` 时才失配——
//     本实现全程 `[32,1]`,不触发;也无需 CUBE #311 TREDUCEPREFIXVIEW。
//   - fp4/hif4 输出走 RowMajor(新 LLVM c9d40c88 gate 掉了 CUBE_M32 的 fp4 B.DATR,
//     `B.DATR CUBE_M32, {HiF4x2,e1m2x2}` Match Instruction Error;RowMajor fp4 可发射,dmxq 同款)。
//   - scale word:走 **uint16 域**(bf16→uint16 同宽,规避 narrowing→u32 的 #119 physical-row
//     契约),lo/hi 两个 uint16 拼成 U32/块,分两半 store。
//
// 结构:abs→三级相邻 max(16 组 [32,4] standalone 载 + TROWMAX + TMAX 树)→ SF=Vmax/7
//   → bf16→e6m2 位重构(问题3)→ M2-LUT 一步倒数(问题4;⚠ cvt+recip 两次舍入错,理想融合
//   cvt_rcpe6m2)→ L2/L3(TCMPS<GE>+TSEL)→ 带符号归一化(每 4-col 组重载+TROWEXPANDMUL
//   +TCVT→fp4+store)→ scale word(uint16 lo/hi)。y≡e1m2(hif4x2 正名待 CUBE fp4 gate 放开)。
// ============================================================================

namespace supernpu::tile_isa::hif4quant {

using namespace pto;

constexpr uint16_t HIF4_INV7_B   = 0x3E12; // 1/7
constexpr uint16_t HIF4_ONE_B    = 0x3F80; // 1.0
constexpr uint16_t HIF4_HALF_B   = 0x3F00; // 0.5
constexpr uint16_t HIF4_THR_L2_B = 0x4080; // 4.0
constexpr uint16_t HIF4_THR_L3_B = 0x4000; // 2.0
inline __bf16 hif4_bf16c(uint16_t b) { return __builtin_bit_cast(__bf16, b); }
constexpr uint16_t HIF4_RECIP_LUT[4] = {0x3F80, 0x3F4D, 0x3F2B, 0x3F12};

// ROW_MAJOR 列向量 [32,1] / uint16 [32,1]
using Row    = Tile<Location::Vec, __bf16,    32, 1, BLayout::RowMajor>;
using U16Row = Tile<Location::Vec, uint16_t,  32, 1, BLayout::RowMajor>;

// ============================================================================
// bf16 → e6m2 8bit 位重构(问题3)⚠ 占位式 RNE
// ============================================================================
inline void bf16_to_e6m2_bits(Row &sf, Row &e6m2_out) {
    auto u = reinterpret_tile<uint16_t>(sf);
    Row mant = sf; auto mu = reinterpret_tile<uint16_t>(mant);
    TSHRS(mu, mu, (uint16_t)5); TANDS(mu, mu, (uint16_t)0x3);
    TSHRS(u, u, (uint16_t)7);   TANDS(u, u, (uint16_t)0xFF);
    TADDS(u, u, (uint16_t)(0x10000 - 79)); TSHLS(u, u, (uint16_t)2); TOR(u, u, mu);
    e6m2_out = sf;
}

// ============================================================================
// e6m2 倒数 → bf16(M2-LUT 位重构)⚠ LUT 选择占位式
// ============================================================================
// ⚠ 占位:rec_bits = LUT[0] + ((48-exp6)<<7),忽略 m2 尾数(仅指数项)。精确 4 选 1
//    LUT[m2] 需 TCMPS 产谓词再 TSEL(留待数值打通阶段);当前 emit/run 见证足够,数值近似。
inline void e6m2_recip_bf16(Row &e6m2, Row &rec_out) {
    rec_out = e6m2; auto ru = reinterpret_tile<uint16_t>(rec_out);
    TSHRS(ru, ru, (uint16_t)2); TANDS(ru, ru, (uint16_t)0x3F);   // exp6
    Row c48; { auto c = reinterpret_tile<uint16_t>(c48); TEXPANDS(c, (uint16_t)48); }
    auto c48u = reinterpret_tile<uint16_t>(c48);
    TSUB(ru, c48u, ru); TSHLS(ru, ru, (uint16_t)7);              // (48-exp6)<<7
    TADDS(ru, ru, (uint16_t)HIF4_RECIP_LUT[0]);                  // + LUT[0](m2=0 近似)
}

// ============================================================================
// (t ≥ K) ? 0.5 : 1.0(bf16 因子)+ E1 位(uint16 0/1)—— TCMPS<GE> + TSEL(整数域)
// ============================================================================
// ⚠ 模型要求 TSEL 的 tuple dtype 为**整数**(IsLogicalIntegerTeplDataType)。故因子选择
//    在 uint16 **位模式**上做(0.5=0x3F00,1.0=0x3F80,选完 reinterpret 回 bf16 即正确值),
//    E1 位直接选整数 0/1。与 dmxq 同款(TSEL 走 uint16)。
template <uint16_t Kbits>
inline void ge_factor_and_bit(Row &t, Row &fac_out, U16Row &e1_out) {
    Row pred; TCMPS<pto::CmpMode::GE>(pred, t, hif4_bf16c(Kbits));
    // 因子:fac_out(bf16)位模式 = pred ? 0.5 : 1.0
    auto fo = reinterpret_tile<uint16_t>(fac_out); TEXPANDS(fo, HIF4_ONE_B);
    Row halfbf; auto ho = reinterpret_tile<uint16_t>(halfbf); TEXPANDS(ho, HIF4_HALF_B);
    TSEL(fo, pred, ho);
    // E1 位(uint16 整数):pred ? 1 : 0
    TEXPANDS(e1_out, (uint16_t)0);
    U16Row one; TEXPANDS(one, (uint16_t)1);
    TSEL(e1_out, pred, one);
}

// ============================================================================
// kernel(V1 蓝本,ROW_MAJOR,BS=64)
// ============================================================================
template <int M, int N, int BlockSize = 64, typename OutT = __fp4_e1m2x2,
          typename InT = __bf16>
void dynamic_hi_f4_quant_tail(InT *x, OutT *y, uint32_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(BlockSize == 64, "hi_f4 BlockSize is fixed 64");
    static_assert(N % BlockSize == 0, "N must be a multiple of 64");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half>,
                  "InT must be __bf16 or __half");

    constexpr int numKb = N / BlockSize;
    constexpr int HALF  = BlockSize / 2;
    constexpr int TileM = (M < 32) ? M : 32;   // 一次处理 TileM 行(≤32)
    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    auto process_tile = [&](int row0) {
        using GrpIn = Tile<Location::Vec, InT,    TileM, 4, BLayout::RowMajor>;   // [TileM,4]
        using GrpBf = Tile<Location::Vec, __bf16, TileM, 4, BLayout::RowMajor>;
        using OutGrp= Tile<Location::Vec, OutT,   TileM, 4, BLayout::RowMajor>;
        using RowT  = Tile<Location::Vec, __bf16, TileM, 1, BLayout::RowMajor>;
        using U16T  = Tile<Location::Vec, uint16_t, TileM, 1, BLayout::RowMajor>;

      for (int kb = 0; kb < numKb; ++kb) {
        // --- 三级相邻 max:16 组 [TileM,4] → TROWMAX [TileM,1];TMAX 树 ---
        RowT m16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            global_tensor<InT, RowMajor<TileM, 4>> gg(x + row0 * N + kb * BlockSize + g * 4);
            GrpIn xg; TLOAD(xg, gg);
            GrpBf ag;
            if constexpr (std::is_same_v<InT, __bf16>) { TABS(ag, xg); }
            else { GrpBf t; TCVT(t, xg); TABS(ag, t); }
            TROWMAX(m16[g], ag);
        }
        RowT m8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) TMAX(m8[g], m16[2 * g], m16[2 * g + 1]);
        RowT m4[4];
#pragma clang loop unroll(full)
        for (int g = 0; g < 4; ++g) TMAX(m4[g], m8[2 * g], m8[2 * g + 1]);
        RowT mA, mB, vmax;
        TMAX(mA, m4[0], m4[1]); TMAX(mB, m4[2], m4[3]); TMAX(vmax, mA, mB);

        // --- base scale + 倒数 ---
        RowT sf; TMULS(sf, vmax, hif4_bf16c(HIF4_INV7_B));
        RowT e6m2; bf16_to_e6m2_bits(sf, e6m2);
        RowT rec;  e6m2_recip_bf16(e6m2, rec);

        // --- L2 / L3 ---
        RowT f8[8]; U16T e1_8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            RowT t8; TMUL(t8, m8[g], rec);
            ge_factor_and_bit<HIF4_THR_L2_B>(t8, f8[g], e1_8[g]);
        }
        RowT f16[16]; U16T e1_16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            RowT t16; TMUL(t16, m16[g], rec); TMUL(t16, t16, f8[g / 2]);
            ge_factor_and_bit<HIF4_THR_L3_B>(t16, f16[g], e1_16[g]);
        }

        // --- 归一化 + 输出(每 4-col 组:重载 signed + 广播乘 + TCVT fp4 + store)---
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            RowT sg; TMUL(sg, rec, f8[g / 2]); TMUL(sg, sg, f16[g]);
            global_tensor<InT, RowMajor<TileM, 4>> gg(x + row0 * N + kb * BlockSize + g * 4);
            GrpIn xg; TLOAD(xg, gg);
            GrpBf xbf;
            if constexpr (std::is_same_v<InT, __bf16>) { xbf = xg; }
            else { TCVT(xbf, xg); }
            GrpBf zg; TROWEXPANDMUL(zg, xbf, sg);
            OutGrp oq; TCVT(oq, zg);
            global_tensor<OutT, RowMajor<TileM, 4>> gy(
                reinterpret_cast<OutT *>(y_u8 + row0 * (N / 2) + kb * HALF + g * 2));
            TSTORE(gy, oq);
        }

        // --- scale word(uint16 域,规避 #119):lo = e6m2 | E1_8<<8;hi = E1_16 16 位 ---
        U16T lo; { auto e = reinterpret_tile<uint16_t>(e6m2); TCVT(lo, e); TANDS(lo, lo, (uint16_t)0xFF); }
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            U16T b; TSHLS(b, e1_8[g], (uint16_t)(8 + g)); TOR(lo, lo, b);
        }
        U16T hi; TEXPANDS(hi, (uint16_t)0);
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            U16T b; TSHLS(b, e1_16[g], (uint16_t)g); TOR(hi, hi, b);
        }
        // 存两个 uint16 到 scale(U32/块 = lo | hi<<16,小端 = 先 lo 后 hi)。
        uint16_t *s16 = reinterpret_cast<uint16_t *>(scale + row0 * numKb + kb);
        global_tensor<uint16_t, RowMajor<TileM, 1>> gslo(s16);
        global_tensor<uint16_t, RowMajor<TileM, 1>> gshi(s16 + 1);
        TSTORE(gslo, lo); TSTORE(gshi, hi);
      }
    };

    for (int r = 0; r < M; r += TileM) process_tile(r);
}

} // namespace supernpu::tile_isa::hif4quant

#endif // SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
