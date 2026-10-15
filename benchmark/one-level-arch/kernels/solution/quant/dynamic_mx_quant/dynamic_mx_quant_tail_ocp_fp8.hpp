#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp8_detail {

// V4.2 tile 类型集（TileN=256）。命名 t_<dtype>_<shape>_<role>。
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

    using t_in_wide   = VecTileM32<InT,         TileMv, WIDE,  ValidRows, WIDE>;   // 宽输入 [32,256]
    using t_in_blk    = VecTileM32<InT,         TileMv, BS,    ValidRows, BS>;     // 数据切片 [32,32]
    using t_in_red    = VecTileM32<InT,         TileMv, BS,    ValidRows, 1>;      // 归约载体 [32,32/1v]（#311：物理宽=源宽）
    using t_f32_frag  = VecTileM32<float,       TileMv, 1,     ValidRows, 1>;      // fp32 拼装 fragment [32,1] = 1 CELL
    using t_f32_amax  = VecTileM32<float,       TileMv, BLKS,  ValidRows, BLKS>;   // fp32 拼装 parent [32,8]
    using t_f32_mid   = VecTileM32<float,       TileMv, BLKS,  ValidRows, BLKS>;   // fp32 归约中转 [32,8]（非bf16 守卫链）
    using t_f32_norm  = VecTileM32<float,       TileMv, BS,    ValidRows, BS>;     // fp32 数据切片 [32,32]（非bf16 data pass）
    using t_f32_pair  = VecTileM32<float,       TileMv, 2,     ValidRows, 2>;      // fp32 pair [32,2]（非bf16 广播）
    using t_bf16_wide = VecTileM32<__bf16,      TileMv, BLKS,  ValidRows, BLKS>;   // bf16 宽链 [32,8]（守卫/recip）
    using t_bf16_pair = VecTileM32<__bf16,      TileMv, 2,     ValidRows, 2>;      // bf16 pair [32,2]（bf16 广播，1 CELL）
    using t_e8m0_out  = VecTileM32<__fp8_e8m0,  TileMv, BLKS,  ValidRows, BLKS>;   // e8m0 scale 宽输出 [32,8]
    using t_e4m3_out  = VecTileM32<__fp8_e4m3,  TileMv, BS,    ValidRows, BS>;     // e4m3 y 输出 [32,32]
};

// 单个 (TileM 行 × WIDE 列) tile 的完整量化。
//
// ── scale pass：把 BLKS 块的行最大值 hstack 成 (32,BLKS)，让守卫链和 scale
//    存储从「BLKS 次窄」变「1 次宽」（TEPL op 数与 TSTORE 次数各 8×↓）──
//   1. 每块一次 TROWMAX（行最大值按块定义，这步无法合并）；
//   2. TCVT(slot, mw)：读出归约结果 + bf16→fp32 加宽 + 写入 parent 第 blk 槽，
//      一步完成「借 fp32 的 CELL 宽度做拼装」（三种候选格式的排除论证见
//      TileTypes 头注释）；归约族没有 slot-dst 重载，分片写只能由 TCVT 这类
//      带 slot 目的地的 op 承载；
//   3. TASSEMBLY → amax (32,BLKS) fp32；
//   4. fp32 域 TANDS floor → TCVT 回 bf16 域——守卫/scale/recip 的常量与位运算
//      全部定义在 bf16 位型域。floor 必须在转换前的 fp32 域做（golden 截断语义），
//      转换只作用于 2 的幂/0/inf 故恒精确——三种 InT 统一走此路径（等价性不依赖
//      「amax 是 bf16 可表示值」，half/fp32 的全精度 amax 亦成立）。
//   5. 宽域守卫/recip 链 + 单次 (32,BLKS) e8m0 宽存。组装 parent 只被 TEPL
//      消费、从不进 TLSU/AGU——#968 的断言只查 TLOAD/TSTORE，此用法不受阻塞。
//
// ── data pass：逐块 y_j = x_j × rp[:,j]（每行系数广播到块的 32 列）──
//   规范（TROWEXPANDMUL.md，normative）两级寻址：B.SUBVIEW 把广播源绑到宽
//   tile 的某个 CELL（SrcSelect 0/1 双槽），B.DATR.RMode 在 CELL 内按字节
//   偏移选列（"is the unsigned BroadcastByteOffset"）。B.SUBVIEW 以整 CELL
//   为粒度（bf16 单列仅半 CELL）→ rp 按 1 CELL=[32,2] 切成 BLKS/2 个 pair，
//   pair 内两列由 RMode 偏移（0/2）选择。
//   双 subview 重载已由 TileOP #253/#254 修复：数据切片（B.SUBVIEW 0）与
//   广播 CELL 切片（B.SUBVIEW 1）同 bundle 发射，无需物化。
//   ⚠ gfrun/gfsim 对双 subview 形态在运行期拒绝（#989 两层：校验断言 +
//   操作数绑定错位）；RMode 偏移解码亦未修（#979）——精度验证依赖 gfrun
//   本地补丁。
//   y 仍逐块窄存，(32,WIDE) 合并输出等 #968 修复后启用。
//
// ── InT 分支语义 ──
//   · bf16：数据与 recip 同为 bf16 → TROWEXPANDMUL 直乘（零转换）。
//   · half/fp32：recip 是 bf16、数据是 InT → dtype 不匹配。统一转到 fp32 域：
//     TCVT(norm_f32, infrag)（InT→fp32 数据切片物化）+ TCVT(rf_f32, pair)
//     （bf16 pair→fp32 pair），再 TROWEXPANDMUL 在 fp32 域乘。
//     与原 dyn 的区别：原 dyn 的 recip 本身就是 fp32（recip_f = TCVT(recip_bf)），
//     V4 的 recip 是 bf16 宽链直出，所以 half 路径多了一次 bf16→fp32 的 pair
//     转换，但省掉了整个 recip 链的 fp32 加宽/回转（V4 宽链在 bf16 域一步到位）。
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
    constexpr int BLKS = TT::BLKS;

    using gm_x = global_tensor<InT, RowMajor<M, N>>;
    using gm_y = global_tensor<__fp8_e4m3, RowMajor<M, N>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<M, scaleCols>>;

    global_iterator<gm_x, typename TT::t_in_wide> x_iter(x + row0 * N + kb0 * BS);
    auto gx = x_iter(0, 0);
    typename TT::t_in_wide xin; TLOAD(xin, gx);
    typename TT::t_in_wide absx; TABS(absx, xin);
    auto absfrags = TPARTVIEW<typename TT::t_in_blk, 1, BLKS>(absx);
    auto infrags  = TPARTVIEW<typename TT::t_in_blk, 1, BLKS>(xin);

    // ===== scale pass：分块归约 → fp32 拼接 (32,BLKS) → 宽域链 → 宽存 =====
    TileArray<typename TT::t_f32_frag, 1, BLKS> amax_arr;
    for (int blk = 0; blk < BLKS; ++blk) {
        auto absfrag = absfrags[0][blk];
        typename TT::t_in_red mw;
        TROWMAX(mw, absfrag);
        TCVT(amax_arr[0][blk], mw);   // 读结果 + bf16→fp32 加宽 + 写第 blk 槽
    }
    typename TT::t_f32_amax amax8 = TASSEMBLY<typename TT::t_f32_amax>(std::move(amax_arr));

    typename TT::t_bf16_wide mb, sh, ni, nz, ns, k, rp;
    auto amax_u32 = reinterpret_tile<uint32_t>(amax8);
    TANDS(amax_u32, amax_u32, FP32_EXP_MASK);  // 就地 fp32 域 floor（= golden 截断语义）。必须先
    //   floor 后转换：先 TCVT(RNE) 会把 amax ∈ [2^k-2^(k-9), 2^k) 向上舍过 binade，half/fp32
    //   的 scale 差 2 倍（模型 TCVT float→float 默认 RNE，见 CubeEngine resolveTcvTRoundingMode；
    //   golden f32_to_bf16_bits 为截断）。bf16 值 fp32 低位恒 0，此 floor 与原 bf16 域 TANDS 同果。
    //   ⚠ 本形态是 spec 预期写法（B.ASSEMBLE:38「parent 发布进普通相对队列」+ TANDS:40「源为
    //   任意已存在 Local 数值 Tile」）：gfrun 逐字节 PASS，但 gfsim 的 #291 严格校验拒绝
    //   「组装产物作元素级源/目的」而 abort——已提 issue 请求裁决统一（gfsim 侧，待编号）。
    //   过渡规避：TASSEMBLY 结果先 TCVT 拷到普通 tile 再就地 floor（+1 op/tile，git 历史存档）。
    TCVT(mb, amax8);                  // 已是 2 的幂/0/inf → bf16 转换精确（三种 InT 统一）
    auto mbu = reinterpret_tile<uint16_t>(mb);
    auto shu = reinterpret_tile<uint16_t>(sh);
    auto niu = reinterpret_tile<uint16_t>(ni);
    auto nzu = reinterpret_tile<uint16_t>(nz);
    auto nsu = reinterpret_tile<uint16_t>(ns);
    auto ku  = reinterpret_tile<uint16_t>(k);
    auto rpu = reinterpret_tile<uint16_t>(rp);
    TCMPS<CmpMode::NE>(niu, mbu, BF16_EXP_MASK);
    TCMPS<CmpMode::NE>(nzu, mbu, static_cast<uint16_t>(0));
    TMULS(sh, mb, __builtin_bit_cast(__bf16, RECIP_EMAX));
    typename TT::t_e8m0_out sb;
    TCVT(sb, sh);
    global_iterator<gm_s, typename TT::t_e8m0_out> si(
        reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb0);
    auto gsi = si(0, 0);
    TSTORE(gsi, sb);                  // 单次 (32,BLKS) e8m0 宽存

    TEXPANDS(ku, BF16_EXP_BIAS);
    TSUB(rpu, ku, shu);
    TCMPS<CmpMode::NE>(nsu, shu, BF16_EXP_BIAS);
    TSELS(rpu, niu, BF16_NAN_PATTERN, rpu);
    TSELS(rpu, nzu, static_cast<uint16_t>(0), rpu);
    TSELS(rpu, nsu, BF16_SPECIAL_EXP, rpu);

    // ===== data pass：recip 切 pair（1 CELL）→ 物化 + RMode 偏移（V4.1 形态）=====
    //   双 subview 直用待 #989 修复后启用（当前 gfrun/gfsim 均拒）。
    auto rppairs = TPARTVIEW<typename TT::t_bf16_pair, 1, BLKS / 2>(rp);
    for (int p = 0; p < BLKS / 2; ++p) {
        auto pairv = rppairs[0][p];
        typename TT::t_bf16_pair pair;     // 物化（mixed 重载要求 src1 为普通 tile）
        TCVT(pair, pairv);
        auto infrag0 = infrags[0][2 * p];
        auto infrag1 = infrags[0][2 * p + 1];
        typename TT::t_e4m3_out oq0, oq1;
        if constexpr (std::is_same_v<InT, __bf16>) {
            typename TT::t_in_blk norm0, norm1;
            TROWEXPANDMUL<typename TT::t_in_blk, typename TT::t_in_wide,
                          typename TT::t_in_blk, typename TT::t_bf16_pair, 0>(
                norm0, infrag0, pair);
            TROWEXPANDMUL<typename TT::t_in_blk, typename TT::t_in_wide,
                          typename TT::t_in_blk, typename TT::t_bf16_pair, 2>(
                norm1, infrag1, pair);
            TCVT(oq0, norm0);
            TCVT(oq1, norm1);
        } else {
            typename TT::t_f32_pair rf;
            TCVT(rf, pairv);              // bf16 pair → fp32 pair
            typename TT::t_f32_norm nf0, nf1;
            TCVT(nf0, infrag0);           // InT 数据切片 → fp32（物化）
            TCVT(nf1, infrag1);
            TROWEXPANDMUL<typename TT::t_f32_norm, typename TT::t_f32_norm,
                          typename TT::t_f32_pair, 0>(
                nf0, nf0, rf);
            TROWEXPANDMUL<typename TT::t_f32_norm, typename TT::t_f32_norm,
                          typename TT::t_f32_pair, 4>(
                nf1, nf1, rf);
            TCVT(oq0, nf0);
            TCVT(oq1, nf1);
        }
        global_iterator<gm_y, typename TT::t_e4m3_out> yb0(
            y + row0 * N + (kb0 + 2 * p) * BS);
        auto gyb0 = yb0(0, 0);
        TSTORE(gyb0, oq0);
        global_iterator<gm_y, typename TT::t_e4m3_out> yb1(
            y + row0 * N + (kb0 + 2 * p + 1) * BS);
        auto gyb1 = yb1(0, 0);
        TSTORE(gyb1, oq1);
    }
}

} // namespace tail_ocp_fp8_detail

// ===========================================================================
// TAIL-OCP-FP8 正式 kernel（V4.2：TileN=256 + 拼装宽链 + 双 subview 直用）
//   固定 SPMD 4-PE。InT(bf16/half/fp32) in / e4m3 out / e8m0 scale / BS=32。
//   每 (32 行 × 256 列) tile：1 次宽 TLOAD(512B/行) + 整块 TABS + 分块归约 →
//   借 fp32 CELL 宽度拼装成 (32,8) → 守卫/recip 全宽域单链 → 单次 (32,8) scale
//   宽存；data pass 双 subview 广播逐块直乘 → 8 × (32,32) y 窄存（合并待 #968）。
//   设计动机与全部约束见 process_tile 注释。
//
//   实测（bench_small [64,16384] bf16，4-PE，V4.1 pair 物化版）：gfrun 精度
//   output/scale 全过（scale 逐字节）；gfsim 54,441 cyc（V4 基座 87,018，-37%）。
//   V4.2 双 subview 版编译通过但 gfrun/gfsim 挂 #989（等上游修复后验证）。
//   CellReg 峰值 57%（TileN 上限调研：顶格 512，1024 全池耗尽）。
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
