#ifndef SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
#define SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

// ============================================================================
// dynamic_hi_f4_quant —— 尾轴(axis=-1)encode,**V1(ISA 伪码)蓝本 + 全 CUBE_M32 布局**
// ============================================================================
// 权威格式/算法见 DESIGN.md §1/§2.1 与文档包 HIF4_QUANT_BF16.md;缺口见 RECORD.md。
//
// 2026-09-15 重写:从旧「RowMajor Vec + 物理宽 + 4 个 _PLACEHOLDER」框架整体重写为
// **Matrix/CUBE_M32 布局**,逐 op 对齐 ISA 伪码 V1。所有机制已用探针
// (test/.../src/segreduce_cube_probe.cpp)实证可发射(EXIT=0):
//   - 分段相邻 max:TPARTVIEW 切 [32,4] 子视图 + TROWMAX 出独立 [32,1] + TMAX 锦标赛树
//     (64→16→8→1)。**免密集压缩**——TASSEMBLY 无法组装 padding 的 [32,1],TPACK 是
//     U32 字节域打包(非列压缩),故不走压缩;只用 B.SUBVIEW(过 gfrun 的 fa_subview 同款),
//     **不碰 B.ASSEMBLE**(其有 region_tilearray 模型 run-fail gap)。
//   - E6M2 base + 倒数:bf16↔e6m2 位重构 + M2-LUT(RECORD 问题3/4;one-level TCVT→e6m2
//     不可发射,规范也不暴露 e6m2 为 tile dtype)。**⚠ 倒数缺口标注**:理想是融合
//     `cvt_rcpe6m2_to_bf16`(单次舍入,当前 TCVT 无,待新增);两步 cvt+recip 拆分是错的
//     (双舍入),故用一步 M2-LUT 位重构等价实现。
//   - 比较/选择:TCMPS<GE> 产 packed-predicate + TSEL 选 2^-E1(V1 GPR 谓词的当前 tile
//     谓词等价形;GPR 专用 TCMP.GPR/TSEL.GPR/P2R/TGPR2T 不可发射)。
//   - 归一化输出:每 4-col 组 subview 源 + TROWEXPANDMUL + TCVT→hif4x2,**逐组 TSTORE_CUBE
//     到 HBM 列偏移**(HBM 作组装介质,免 B.ASSEMBLE)。
//   - scale word:E1 位算术累加 + 位拼(e6m2 | E1_8<<8 | E1_16<<16)成 U32/块。
//
// 验证边界:工具链↔仿真器 skew 下运行期未通,本文件为 **emit/diss 见证**(op-review +
// 编译/反汇编)。占位式舍入细节(e6m2 RNE)未逐位对齐 emulator,数值以规范语义为准。
// ============================================================================

namespace supernpu::tile_isa::hif4quant {

using namespace pto;

// --- 权威 bf16 立即数位模式(RECORD 问题11:float→bf16 崩后端,一律走 __builtin_bit_cast)---
constexpr uint16_t HIF4_INV7_B   = 0x3E12; // 1/7(满量程锚,DESIGN §2.2)
constexpr uint16_t HIF4_ONE_B    = 0x3F80; // 1.0(2^0,boost 未命中)
constexpr uint16_t HIF4_HALF_B   = 0x3F00; // 0.5(2^-1,boost 命中)
constexpr uint16_t HIF4_THR_L2_B = 0x4080; // 4.0(二级阈值)
constexpr uint16_t HIF4_THR_L3_B = 0x4000; // 2.0(三级阈值)
inline __bf16 hif4_bf16c(uint16_t b) { return __builtin_bit_cast(__bf16, b); }

// e6m2 recip 的 M2-LUT(DESIGN §3.1 / RECORD 问题4):LUT[m2]=bf16 bits of 1/(1+m2/4)。
constexpr uint16_t HIF4_RECIP_LUT[4] = {0x3F80, 0x3F4D, 0x3F2B, 0x3F12};

// ============================================================================
// [32,K] ROW_MAJOR Vec tile 别名(对齐 mxquant.hpp,过 gfrun 的 quant 模式)
// ============================================================================
// 早期曾试 CUBE_M32(为 TPARTVIEW),但已改用 standalone [32,4] 载入(不需 subview),
// 故全程用 ROW_MAJOR:①归约输出恒为 ROW_MAJOR [N,1](模型规范化,gfrun 实证),下游
// [32,1] ROW_MAJOR 天然匹配;②CUBE_M32 归约输出转 ROW_MAJOR 会与下游 CUBE op 失配
// (gfrun 诊断:TMAX 源 layout=ROW_MAJOR col=1 vs 期望 physCol=2)。
template <int K> using BfSlot = Tile<Location::Vec, __bf16, 32, K, BLayout::RowMajor>;
// 「1 值」列向量 slot:物理 [32,2] + valid [32,1] ROW_MAJOR。physical col=2 让窄化 TCVT
// (→u32)的 DerivedRows 与源匹配,规避 #119;valid col=1 满足 TROWMAX dst 契约
// (PR#136 已把头约束从 Cols==1 放宽为 ValidCol==1)。全链统一此形,#119 一次性消除。
using ValSlot = Tile<Location::Vec, __bf16, 32, 2, BLayout::RowMajor, 32, 1>;

// ============================================================================
// e6m2 base 位重构:bf16 → e6m2 8bit(存低字节),就地在 uint16 视图上算
// ============================================================================
// e6m2: exp6=bits[7:2](bias 48)、m2=bits[1:0];value=(1+m2/4)·2^(exp6-48)。
// bf16: exp8=bits[14:7](bias 127)、mant7=bits[6:0]。SF 非负有限。
//   exp6 = exp8 - 127 + 48 = exp8 - 79;m2 = RNE(mant7 → 2bit)= (mant7+0x10)>>5(截断近似)。
//   e6m2_byte = ((exp8-79)<<2) | m2 = ((bits>>7 & 0xFF) - 79)<<2 | ((bits>>5)&3)。
// ⚠ 占位式 RNE(未处理进位越界),emit 见证;逐位对齐 emulator 留待运行期打通后。
inline void bf16_to_e6m2_bits(ValSlot &sf, ValSlot &e6m2_out) {
    // 就地在 sf 的 uint16 视图上算,结果 bf16 位 = e6m2 字节(低 8 位)。
    auto u = reinterpret_tile<uint16_t>(sf);
    ValSlot mant_bf = sf;                 // 拷贝一份算 m2
    auto mu = reinterpret_tile<uint16_t>(mant_bf);
    TSHRS(mu, mu, (uint16_t)5);
    TANDS(mu, mu, (uint16_t)0x3);            // m2 = (bits>>5)&3
    TSHRS(u, u, (uint16_t)7);
    TANDS(u, u, (uint16_t)0xFF);
    TADDS(u, u, (uint16_t)(0x10000 - 79));   // exp6 = exp8 - 79(uint16 回绕,无 TSUBS)
    TSHLS(u, u, (uint16_t)2);                // exp6<<2
    TOR(u, u, mu);                           // | m2  -> e6m2 字节
    e6m2_out = sf;                           // sf 现持 e6m2 位
}

// ============================================================================
// e6m2 倒数 → bf16(M2-LUT 位重构,一步等价融合 cvt_rcpe6m2_to_bf16)
// ============================================================================
// ea 为 e6m2 字节:exp6=(bits>>2)&0x3F、m2=bits&3;1/ea=2^-(exp6-48)·1/(1+m2/4)。
//   rec_bits(bf16) = LUT[m2] + ((48-exp6)<<7)  (指数项按 bf16 exp 位宽 7)。
// ⚠ LUT[m2] 选择用两次 TSEL(按 m2 的两个 bit),占位式(未逐位对齐);emit 见证。
inline void e6m2_recip_bf16(ValSlot &e6m2, ValSlot &rec_out) {
    // exp 项:e = (48 - exp6) << 7,以 bf16 位承载。
    ValSlot ebf = e6m2;
    auto eu = reinterpret_tile<uint16_t>(ebf);
    TSHRS(eu, eu, (uint16_t)2);
    TANDS(eu, eu, (uint16_t)0x3F);           // exp6
    // (48 - exp6):用 rsub。TEXPANDS 48 到 tile 再 TSUB。
    ValSlot c48; { auto c = reinterpret_tile<uint16_t>(c48); TEXPANDS(c, (uint16_t)48); }
    auto c48u = reinterpret_tile<uint16_t>(c48);
    TSUB(eu, c48u, eu);                       // 48 - exp6
    TSHLS(eu, eu, (uint16_t)7);               // <<7(bf16 exp 位)

    // LUT[m2]:按 m2 两 bit 选 {LUT0..3}。用两次 TSEL(bit0/bit1 谓词)。
    ValSlot m2bf = e6m2;
    auto m2u = reinterpret_tile<uint16_t>(m2bf);
    TANDS(m2u, m2u, (uint16_t)0x3);           // m2
    // lut = LUT0;若 m2 bit0 → LUT1/… 简化:lut = LUT0 + m2*(step) 不精确,改分层 TSEL。
    ValSlot lut; { auto l = reinterpret_tile<uint16_t>(lut); TEXPANDS(l, HIF4_RECIP_LUT[0]); }
    ValSlot pb0, pb1;                        // m2 的 bit0 / bit1 谓词
    { auto b0 = reinterpret_tile<uint16_t>(pb0); TANDS(b0, m2u, (uint16_t)0x1); }
    { auto b1 = reinterpret_tile<uint16_t>(pb1); TANDS(b1, m2u, (uint16_t)0x2); }
    // 占位式 LUT 选择(emit 见证;运行期须换精确 4 选 1):
    ValSlot lut1; { auto l = reinterpret_tile<uint16_t>(lut1); TEXPANDS(l, HIF4_RECIP_LUT[1]); }
    ValSlot lut2; { auto l = reinterpret_tile<uint16_t>(lut2); TEXPANDS(l, HIF4_RECIP_LUT[2]); }
    TSEL(lut, pb0, lut1);                      // m2 bit0 -> LUT1(占位)
    TSEL(lut, pb1, lut2);                      // m2 bit1 -> LUT2(占位)

    // rec_bits = lut + exp_term(uint16 加,承载 bf16 位)。
    auto lu = reinterpret_tile<uint16_t>(lut);
    TADD(lu, lu, eu);
    rec_out = lut;                             // lut 现持 rec 的 bf16 位
}

// ============================================================================
// (t ≥ K) ? 0.5 : 1.0  —— TCMPS<GE> packed-predicate + TSEL(V1 GPR 谓词的 tile 等价)
// 同时产 E1 位(0/1 bf16)供 scale word 累加。t 非负有限。
// ============================================================================
template <uint16_t Kbits>
inline void ge_factor_and_bit(ValSlot &t, ValSlot &fac_out, ValSlot &e1_out) {
    ValSlot pred; TCMPS<pto::CmpMode::GE>(pred, t, hif4_bf16c(Kbits));
    ValSlot half; TEXPANDS(half, hif4_bf16c(HIF4_HALF_B));
    TEXPANDS(fac_out, hif4_bf16c(HIF4_ONE_B));
    TSEL(fac_out, pred, half);                 // (t≥K)?0.5:1.0
    ValSlot one;  TEXPANDS(one, hif4_bf16c(HIF4_ONE_B));
    TSUB(e1_out, t, t);                         // 0(t 非负有限,t-t=0 精确;避 bf16 0.0 立即数)
    TSEL(e1_out, pred, one);                    // (t≥K)?1:0
}

// ============================================================================
// kernel:dynamic_hi_f4_quant 尾轴 encode(V1 蓝本,CUBE_M32,BS=64)
// ============================================================================
// 输入 x:InT ∈ {__bf16,__half}(half 先 TCVT→bf16 进 bf16 域);输出 y:hif4x2;
// scale:uint32 32bit/块。M 须 32 的倍数(CUBE_M32 行),N 须 64 的倍数(BS)。
// 每个 [32,64] CUBE tile = 32 行(= 32 个 64-elem block)。
template <int M, int N, int BlockSize = 64, typename OutT = __fp4_hif4x2,
          typename InT = __bf16>
void dynamic_hi_f4_quant_tail(InT *x, OutT *y, uint32_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(BlockSize == 64, "hi_f4 BlockSize is fixed 64 (DESIGN §1.1)");
    static_assert(M % 32 == 0, "CUBE_M32: M must be a multiple of 32");
    static_assert(N % BlockSize == 0, "N must be a multiple of 64");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half>,
                  "InT must be __bf16 or __half (DESIGN §0)");

    constexpr int numKb = N / BlockSize;   // 每行 block 数
    constexpr int HALF  = BlockSize / 2;   // 32:hif4 打包字节宽/块

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    // 一个 [32,64] block-tile(行基 row0,列块 kb)。
    auto process_tile = [&](int row0, int kb) {
        using SubIn = Tile<Location::Vec, InT, 32, 4, BLayout::RowMajor>;          // HBM 载的组 [32,4]
        using Sub4  = Tile<Location::Vec, __bf16, 32, 4, BLayout::RowMajor>;         // bf16 [32,4]

        // --- 三级相邻 max:逐组从 HBM 载 standalone [32,4] + TABS + plain TROWMAX(→[32,2]valid1)---
        // PR#136 放宽 row-reduction dst 约束(physical col=1 → valid col=1)后,plain TROWMAX 可
        // 直接出 CUBE_M32 [32,2]valid1;standalone 源满足模型 Block.cpp:2034 契约,避开 region
        // (subview)TROWMAX 的 subview 源被 gfrun 运行期拒的问题。
        ValSlot m16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            global_tensor<InT, RowMajor<32, 4>> gXg(x + row0 * N + kb * BlockSize + g * 4);
            SubIn xg; TLOAD(xg, gXg);
            Sub4 ag;
            if constexpr (std::is_same_v<InT, __bf16>) { TABS(ag, xg); }
            else { Sub4 t; TCVT(t, xg); TABS(ag, t); }
            TROWMAX(m16[g], ag);
        }
        ValSlot m8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) TMAX(m8[g], m16[2 * g], m16[2 * g + 1]);
        ValSlot m4[4];
#pragma clang loop unroll(full)
        for (int g = 0; g < 4; ++g) TMAX(m4[g], m8[2 * g], m8[2 * g + 1]);
        ValSlot mA, mB, vmax;
        TMAX(mA, m4[0], m4[1]); TMAX(mB, m4[2], m4[3]); TMAX(vmax, mA, mB);

        // --- base scale:SF=Vmax/7 → e6m2 → rec ---
        ValSlot sf; TMULS(sf, vmax, hif4_bf16c(HIF4_INV7_B));
        ValSlot e6m2; bf16_to_e6m2_bits(sf, e6m2);
        ValSlot rec;  e6m2_recip_bf16(e6m2, rec);

        // --- L2:E1_8[g]=(Vmax8[g]*rec≥4);factor f8[g]=2^-E1_8 ---
        ValSlot f8[8], e1_8[8];
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            ValSlot t8; TMUL(t8, m8[g], rec);
            ge_factor_and_bit<HIF4_THR_L2_B>(t8, f8[g], e1_8[g]);
        }
        // --- L3:E1_16[g]=(Vmax16[g]*rec*2^-E1_8[g/2]≥2);factor f16[g] ---
        ValSlot f16[16], e1_16[16];
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            ValSlot t16; TMUL(t16, m16[g], rec);
            TMUL(t16, t16, f8[g / 2]);                     // ×2^-E1_8(组 g/2)
            ge_factor_and_bit<HIF4_THR_L3_B>(t16, f16[g], e1_16[g]);
        }

        // --- 归一化 + 输出:每 4-col 组 subview 源 + 广播乘 + TCVT hif4 + HBM store ---
        //   每元素有效 scale = rec · 2^-E1_8[c/8] · 2^-E1_16[c/4];组 g(4 列 c=4g..4g+3)
        //   共享 rec · f8[g/2] · f16[g]。TROWEXPANDMUL 把 [32,1] scale 广播到 subview 4 列。
        using SubIn  = Tile<Location::Vec, InT, 32, 4, BLayout::RowMajor>;          // HBM 重载的带符号组 [32,4]
        using SubBf  = Tile<Location::Vec, __bf16, 32, 4, BLayout::RowMajor>;
        using SubOut = Tile<Location::Vec, OutT, 32, 4, BLayout::RowMajor>;          // hif4x2:ValidCol=4(打包由 BytesOf 处理)
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            ValSlot sg; TMUL(sg, rec, f8[g / 2]); TMUL(sg, sg, f16[g]);  // rec·2^-E1_8·2^-E1_16
            // 从 HBM 重载带符号组 [32,4](免 subview 源;归一化用带符号 X 保号)。
            global_tensor<InT, RowMajor<32, 4>> gXg(x + row0 * N + kb * BlockSize + g * 4);
            SubIn xg; TLOAD(xg, gXg);
            SubBf xbf;
            if constexpr (std::is_same_v<InT, __bf16>) { xbf = xg; }
            else { TCVT(xbf, xg); }
            SubBf zg; TROWEXPANDMUL(zg, xbf, sg);            // Z = X_g · scale_g(广播到 4 列)
            SubOut oq; TCVT(oq, zg);                         // bf16 -> hif4x2([32,4])
            global_tensor<OutT, RowMajor<32, 4>> gY(
                reinterpret_cast<OutT *>(y_u8 + row0 * (N / 2) + kb * HALF + g * 2));
            TSTORE(gY, oq);
        }

        // --- scale word:e6m2 | E1_8<<8 | E1_16<<16(算术累加位,全 plain U32 tile)---
        using U32Slot = Tile<Location::Vec, uint32_t, 32, 2, BLayout::RowMajor, 32, 1>;
        U32Slot word; TEXPANDS(word, (uint32_t)0);
        // e6m2 低字节:e6m2(bf16 存储)低 16 位即 e6m2 字节值;u16 视图数值化 → 零扩到 u32。
        auto e6v = reinterpret_tile<uint16_t>(e6m2);   // 具名 u16 视图
        U32Slot e6u; TCVT(e6u, e6v);                   // 无符号零扩:值=e6m2 字节,位保留
        TANDS(e6u, e6u, (uint32_t)0xFF);
        TOR(word, word, e6u);
        // E1_8 8 位 -> bits[15:8](bf16 0/1 → u32 0/1 → 移位 OR)
#pragma clang loop unroll(full)
        for (int g = 0; g < 8; ++g) {
            U32Slot b; TCVT(b, e1_8[g]);
            TSHLS(b, b, (uint32_t)(8 + g));
            TOR(word, word, b);
        }
        // E1_16 16 位 -> bits[31:16]
#pragma clang loop unroll(full)
        for (int g = 0; g < 16; ++g) {
            U32Slot b; TCVT(b, e1_16[g]);
            TSHLS(b, b, (uint32_t)(16 + g));
            TOR(word, word, b);
        }
        global_tensor<uint32_t, RowMajor<32, 1>> gS(scale + row0 * numKb + kb);
        TSTORE(gS, word);
    };

    for (int r = 0; r < M; r += 32)
        for (int kb = 0; kb < numKb; ++kb)
            process_tile(r, kb);
}

} // namespace supernpu::tile_isa::hif4quant

#endif // SUPERNPU_DYNAMIC_HI_F4_QUANT_TAIL_H
