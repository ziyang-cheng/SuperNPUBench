#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp8_detail {

// V4.1 tile 类型集（TileN=256）。
//   拼装格式定为 fp32，是 fragment 与 parent 两侧合同同时可满足的唯一解：
//   · bf16 [32,1] fragment = 64B（半 CELL）：slot 偏移按 128B CELL 编码，
//     fragment 必须整 CELL——直接被拒；
//   · bf16 [32,2] fragment = 128B ✓：但 parent 被迫为 [32,16]phys/[32,8]valid，
//     而 #291 要求 CUBE 元素级源的存储列 = align(validCol, CELL 列数) =
//     align(8, 2) = 8——[32,16] 违约，宽链第一条 op 即被拒；
//   · fp32 [32,1] fragment = 128B ✓，parent [32,8]phys/[32,8]valid，
//     #291 期望 align(8, 1) = 8 ✓。
//   故拼装借 fp32 的 CELL 宽度、拼完 TCVT 转回原域（两次转换均无损，
//   见 process_tile 注释）。
template <typename InT, int TileMv, int ValidRows, int WIDE, int BS>
struct TileTypes {
    static constexpr int BLKS = WIDE / BS;   // 每 tile 的 MX 块数
    static_assert(BLKS % 2 == 0, "pair slicing requires an even block count");

    using t_blk = VecTileM32<InT,        TileMv, WIDE,  ValidRows, WIDE>;   // 宽输入
    using t_cut = VecTileM32<InT,        TileMv, BS,    ValidRows, BS>;     // 数据切片
    using t_rw  = VecTileM32<InT,        TileMv, BS,    ValidRows, 1>;      // 归约载体（#311：物理宽=源宽）
    using t_af  = VecTileM32<float,      TileMv, 1,     ValidRows, 1>;      // 拼装 fragment：fp32 [32,1] = 1 CELL
    using t_a8  = VecTileM32<float,      TileMv, BLKS,  ValidRows, BLKS>;   // 拼装 parent：amax (32,BLKS) fp32
    using t_b8  = VecTileM32<__bf16,     TileMv, BLKS,  ValidRows, BLKS>;   // bf16 域宽链（守卫/recip）
    using t_e8w = VecTileM32<__fp8_e8m0, TileMv, BLKS,  ValidRows, BLKS>;   // e8m0 宽输出
    using t_rp2 = VecTileM32<__bf16,     TileMv, 2,     ValidRows, 2>;      // recip pair：[32,2] = 1 CELL（B.SUBVIEW 最小粒度）
    using t_fr2 = VecTileM32<float,      TileMv, 2,     ValidRows, 2>;      // fp32 pair（非 bf16 输入的转换）
    using t_o   = VecTileM32<__fp8_e4m3, TileMv, BS,    ValidRows, BS>;     // y 输出
};

// 单个 (TileM 行 × WIDE 列) tile 的完整量化。
//
// ── scale pass：把 BLKS 块的行最大值 hstack 成 (32,BLKS)，让守卫链和 scale
//    存储从「BLKS 次窄」变「1 次宽」（TEPL op 数与 TSTORE 次数各 8×↓）──
//   1. 每块一次 TROWMAX（行最大值按块定义，这步无法合并）；
//   2. TCVT(slot, mw)：读出归约结果 + bf16→fp32 加宽 + 写入 parent 第 blk 槽，
//      一步完成「借 fp32 的 CELL 宽度做拼装」（三种候选格式的排除论证见
//      TileTypes 头注释：bf16 [32,1] 半 CELL、bf16 [32,2] 撑破 #291 的
//      parent 存储合同，唯 fp32 [32,1] 两侧皆可）；归约族没有 slot-dst
//      重载，分片写只能由 TCVT 这类带 slot 目的地的 op 承载；
//   3. TASSEMBLY → amax8 (32,BLKS) fp32；
//   4. TCVT(mb, amax8)：fp32→bf16 转回原精度域——守卫/scale/recip 的常量与
//      位运算全部定义在 bf16 位型域。bf16→fp32→bf16 往返逐位精确（amax 本就
//      是 bf16 可表示的值），故数值路径与逐块窄链版完全等价；
//   5. 宽域守卫/recip 链 + 单次 (32,BLKS) e8m0 宽存。组装 parent 只被 TEPL
//      消费、从不进 TLSU/AGU——#968 的断言只查 TLOAD/TSTORE，此用法不受阻塞。
//
// ── data pass：逐块 y_j = x_j × rp[:,j]（每行系数广播到块的 32 列）──
//   规范（docs/tile/reduce-and-expand/row-expansion/TROWEXPANDMUL.md，normative）
//   为「从宽 tile 选一列广播」约定了两级寻址：
//     · B.SUBVIEW 先把广播源绑到宽 tile 的某个 CELL——"B.SUBVIEW selects a
//       later CELL/range when needed"（SrcSelect 0/1 各占一源，双 subview 合法）；
//     · B.DATR.RMode 再在 CELL 内按字节偏移选列——"For CUBE_M32 and CUBE_M16,
//       B.DATR.RMode is the unsigned BroadcastByteOffset and not a numeric
//       rounding selector"；
//     · 广播源允许一个 CELL 打包多列、仅被选列供值——"later valid columns
//       are ignored and are not encoding-validated"。
//   B.SUBVIEW 以整 CELL 为粒度（bf16 单列仅半 CELL）→ rp 按 1 CELL=[32,2]
//   切成 BLKS/2 个 pair，pair 内两列由 RMode 字节偏移（0/2）选择。
//   现状两个工具缺口（上游补齐后本段可再省 4 次 TCVT）：
//   ① TileOP 缺「数据切片 × 广播切片」双 subview 发射重载（现成 both-subview
//     版是 bcast×bcast 语义），可用的 mixed 重载对广播源整寄存器读 → pair
//     须先 TCVT 复制成独立寄存器（每 tile 4 次，纯开销）；
//   ② gfrun 不解码 RMode（违反上述 normative 条款，静默错值）→ 依赖
//     TEPLEngine.cpp 本地补丁按偏移取列。
//   y 仍逐块窄存，(32,WIDE) 合并输出等 #968 修复后启用。
//
// ── 结构约束 ──
//   · 命名函数、裸指针参数：TileArray 在 lambda 内创建会地址逃逸（S32 spill）；
//   · noinline：B.IOT PE mask=1111 使 B.ASSEMBLE session 的 publish 需全部
//     4 个 PE 在同一 INIT pc 完成登记；内联成多份拷贝会让不同 PE 组执行不同
//     pc，session 永不 publish（gfrun FAIL E 实证）。单函数体 = 单 pc。
template <typename InT, int TileMv, int ValidRows, int WIDE, int BS,
          int M, int N, int scaleCols, uint16_t RECIP_EMAX>
__attribute__((noinline))
void process_tile(InT *x, __fp8_e4m3 *y, uint8_t *scale,
                         int64_t row0, int kb0) {
    using namespace pto;
    using TT = TileTypes<InT, TileMv, ValidRows, WIDE, BS>;
    using t_blk = typename TT::t_blk;
    using t_cut = typename TT::t_cut;
    constexpr int BLKS = TT::BLKS;

    using gm_x = global_tensor<InT, RowMajor<M, N>>;
    using gm_y = global_tensor<__fp8_e4m3, RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    global_iterator<gm_x, t_blk> x_iter(x + row0 * N + kb0 * BS);
    auto gx = x_iter(0, 0);
    t_blk xin; TLOAD(xin, gx);
    t_blk absx; TABS(absx, xin);
    auto absfrags = TPARTVIEW<t_cut, 1, BLKS>(absx);
    auto infrags  = TPARTVIEW<t_cut, 1, BLKS>(xin);

    // ===== scale pass：分块归约 → fp32 拼接 (32,BLKS) → 宽域链 → 宽存 =====
    TileArray<typename TT::t_af, 1, BLKS> amax_arr;
    for (int blk = 0; blk < BLKS; ++blk) {
        auto absfrag = absfrags[0][blk];
        typename TT::t_rw mw;
        TROWMAX(mw, absfrag);             // 每块行最大值（按块定义，不可合并）
        TCVT(amax_arr[0][blk], mw);       // 读结果 + bf16→fp32 加宽 + 写第 blk 槽
    }
    typename TT::t_a8 amax8 = TASSEMBLY<typename TT::t_a8>(std::move(amax_arr));

    typename TT::t_b8 mb, sh, ni, nz, ns, k, rp;
    TCVT(mb, amax8);                       // fp32→bf16 回原域（往返精确），守卫链
                                          // 常量与位运算在 bf16 位型域
    auto mbu = reinterpret_tile<uint16_t>(mb);
    auto shu = reinterpret_tile<uint16_t>(sh);
    auto niu = reinterpret_tile<uint16_t>(ni);
    auto nzu = reinterpret_tile<uint16_t>(nz);
    auto nsu = reinterpret_tile<uint16_t>(ns);
    auto ku  = reinterpret_tile<uint16_t>(k);
    auto rpu = reinterpret_tile<uint16_t>(rp);
    TANDS(mbu, mbu, BF16_EXP_MASK);
    TCMPS<CmpMode::NE>(niu, mbu, BF16_EXP_MASK);
    TCMPS<CmpMode::NE>(nzu, mbu, static_cast<uint16_t>(0));
    TMULS(sh, mb, __builtin_bit_cast(__bf16, RECIP_EMAX));
    typename TT::t_e8w sb;
    TCVT(sb, sh);
    global_iterator<gm_s, typename TT::t_e8w> si(
        reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb0);
    auto gsi = si(0, 0);
    TSTORE(gsi, sb);                       // 单次 (32,BLKS) e8m0 宽存

    TEXPANDS(ku, BF16_EXP_BIAS);
    TSUB(rpu, ku, shu);
    TCMPS<CmpMode::NE>(nsu, shu, BF16_EXP_BIAS);
    TSELS(rpu, niu, BF16_NAN_PATTERN, rpu);
    TSELS(rpu, nzu, static_cast<uint16_t>(0), rpu);
    TSELS(rpu, nsu, BF16_SPECIAL_EXP, rpu);

    // ===== data pass：recip 切 pair（1 CELL）→ 打包 CELL 广播 + 字节偏移 =====
    auto rppairs = TPARTVIEW<typename TT::t_rp2, 1, BLKS / 2>(rp);
    for (int p = 0; p < BLKS / 2; ++p) {
        auto pairv  = rppairs[0][p];
        typename TT::t_rp2 pair;      // 物化（mixed 重载要求 src1 为普通 tile）
        TCVT(pair, pairv);
        auto infrag0 = infrags[0][2 * p];
        auto infrag1 = infrags[0][2 * p + 1];
        typename TT::t_cut norm0, norm1;
        if constexpr (std::is_same_v<InT, __bf16>) {
            // RMode 字节偏移选 pair 内列（规范 normative；gfrun 需本地补丁——缺口②）
            TROWEXPANDMUL<typename TT::t_cut, typename TT::t_blk,
                          typename TT::t_cut, typename TT::t_rp2, 0>(
                norm0, infrag0, pair);                // 打包 CELL 广播，列 0
            TROWEXPANDMUL<typename TT::t_cut, typename TT::t_blk,
                          typename TT::t_cut, typename TT::t_rp2, 2>(
                norm1, infrag1, pair);                // 列 1（bf16 字节偏移 2）
        } else {
            typename TT::t_fr2 rf;
            TCVT(rf, pair);
            TCVT(norm0, infrag0);
            TCVT(norm1, infrag1);
            TROWEXPANDMUL<typename TT::t_cut, typename TT::t_cut,
                          typename TT::t_fr2, 0>(norm0, norm0, rf);
            TROWEXPANDMUL<typename TT::t_cut, typename TT::t_cut,
                          typename TT::t_fr2, 4>(norm1, norm1, rf);  // fp32 偏移 4
        }
        typename TT::t_o oq0, oq1;
        TCVT(oq0, norm0);
        TCVT(oq1, norm1);
        global_iterator<gm_y, typename TT::t_o> yb0(
            y + row0 * N + (kb0 + 2 * p) * BS);
        auto gyb0 = yb0(0, 0);
        TSTORE(gyb0, oq0);
        global_iterator<gm_y, typename TT::t_o> yb1(
            y + row0 * N + (kb0 + 2 * p + 1) * BS);
        auto gyb1 = yb1(0, 0);
        TSTORE(gyb1, oq1);
    }
}

} // namespace tail_ocp_fp8_detail

// ===========================================================================
// TAIL-OCP-FP8 正式 kernel（V4.1：TileN=256 + scale 侧拼装宽链）
//   固定 SPMD 4-PE。InT(bf16/half/fp32) in / e4m3 out / e8m0 scale / BS=32。
//   每 (32 行 × 256 列) tile：1 次宽 TLOAD(512B/行) + 整块 TABS + 分块归约 →
//   借 fp32 CELL 宽度拼装成 (32,8) → 守卫/recip 全宽域单链 → 单次 (32,8) scale
//   宽存；data pass 以 pair 打包广播逐块直乘 → 8 × (32,32) y 窄存（合并待 #968）。
//   设计动机与全部约束见 process_tile 注释。
//
//   实测（bench_small [64,16384] bf16，4-PE）：gfrun 精度 output/scale 全过
//   （scale 逐字节）；gfsim 默认配置 54,441 cyc（V4 基座 87,018，-37%）。
//   CellReg 峰值 57%（上限调研：TileN 顶格 512，1024 全池耗尽）。
//
//   ⚠ 运行依赖两处本地补丁（上游缺口，待提 issue）：
//   ① 工具链 TileOP #262：PhysicalCol 1 行修复（已提 issue）；
//   ② gfrun TEPLEngine.cpp：TROWEXPAND 广播字节偏移解码（B.DATR RMode 位）。
// ===========================================================================
template <int M, int N, int BlockSize = 32, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8(InT *x, __fp8_e4m3 *y, uint8_t *scale) {
    static_assert(M > 0 && N > 0, "dim must be positive");
    static_assert(M % 32 == 0,
                  "TASSEMBLY variant requires whole 32-row tiles: B.ASSEMBLE "
                  "cross-PE sessions need a single process_tile site executed "
                  "by all PEs");
    static_assert(N % 256 == 0, "V4 TileN=256: N must be multiple of 256");
    static_assert(BlockSize % 32 == 0, "BlockSize must be multiple of 32");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>();
    constexpr int numKb     = N / BlockSize;
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;
    constexpr int kPeNum    = 4;
    constexpr int WIDE      = 256;
    constexpr int BlocksPerTile = WIDE / BlockSize;  // 8

    const uint32_t tid = get_thread_idx();

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
            if constexpr (myTiles > 0 && myKb > 0) {
                for (int t = 0; t < myTiles; ++t) {
                    for (int kb = kbBegin; kb < kbBegin + myKb; ++kb) {
                        tail_ocp_fp8_detail::process_tile<
                            InT, TileM, TileM, WIDE, BlockSize,
                            M, N, scaleCols, RECIP_EMAX>(
                            x, y, scale, (myTileBegin + t) * TileM,
                            kb * BlocksPerTile);
                    }
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
