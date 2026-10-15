#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

namespace tail_ocp_fp8_dyn_detail {

// V4.2 tile 类型集（TileN=256），镜像静态版 tail_ocp_fp8_detail::TileTypes。
//   命名 t_<dtype>_<shape>_<role>。
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
    using t_f32_norm  = VecTileM32<float,       TileMv, BS,    ValidRows, BS>;     // fp32 数据切片 [32,32]（非bf16 data pass）
    using t_f32_pair  = VecTileM32<float,       TileMv, 2,     ValidRows, 2>;      // fp32 pair [32,2]（非bf16 广播）
    using t_bf16_wide = VecTileM32<__bf16,      TileMv, BLKS,  ValidRows, BLKS>;   // bf16 宽链 [32,8]（守卫/recip）
    using t_bf16_pair = VecTileM32<__bf16,      TileMv, 2,     ValidRows, 2>;      // bf16 pair [32,2]（bf16 广播，1 CELL）
    using t_e8m0_out  = VecTileM32<__fp8_e8m0,  TileMv, BLKS,  ValidRows, BLKS>;   // e8m0 scale 宽输出 [32,8]
    using t_e4m3_out  = VecTileM32<__fp8_e4m3,  TileMv, BS,    ValidRows, BS>;     // e4m3 y 输出 [32,32]
};

// 单个 (TileM 行 × WIDE 列) tile 的完整量化——镜像静态版 process_tile，唯一
//   差异：M/N/scaleCols 为运行期参数（tiling[] 传入），GM 描述符用
//   RowMajor<-1,-1> 运行期构造（指针即 tile 左上角，行跨 = N，只访问首 32 行）。
//
// ── scale pass：把 BLKS 块的行最大值 hstack 成 (32,BLKS)，让守卫链和 scale
//    存储从「BLKS 次窄」变「1 次宽」（TEPL op 数与 TSTORE 次数各 8×↓）──
//   1. 每块一次 TROWMAX（行最大值按块定义，这步无法合并）；
//   2. TCVT(slot, mw)：读出归约结果 + InT→fp32 加宽 + 写入 parent 第 blk 槽
//      （三种候选格式的排除论证见 TileTypes 头注释）；归约族没有 slot-dst
//      重载，分片写只能由 TCVT 这类带 slot 目的地的 op 承载；
//   3. TASSEMBLY → amax (32,BLKS) fp32；
//   4. fp32 域 TANDS floor → TCVT 回 bf16 域——守卫/scale/recip 的常量与位
//      运算全部定义在 bf16 位型域。floor 必须在转换前的 fp32 域做（golden
//      截断语义），转换只作用于 2 的幂/0/inf 故恒精确——三种 InT 统一走此
//      路径（等价性不依赖「amax 是 bf16 可表示值」，half/fp32 的全精度 amax
//      亦成立）。
//   5. 宽域守卫/recip 链 + 单次 (32,BLKS) e8m0 宽存。组装 parent 只被 TEPL
//      消费、从不进 TLSU/AGU——#968 的断言只查 TLOAD/TSTORE，此用法不受阻塞。
//
// ── data pass：逐块 y_j = x_j × rp[:,j]（每行系数广播到块的 32 列）──
//   规范（TROWEXPANDMUL.md，normative）两级寻址：B.SUBVIEW 把广播源绑到宽
//   tile 的某个 CELL（SrcSelect 0/1 双槽），B.DATR.RMode 在 CELL 内按字节
//   偏移选列。B.SUBVIEW 以整 CELL 为粒度（bf16 单列仅半 CELL）→ rp 按
//   1 CELL=[32,2] 切成 BLKS/2 个 pair，pair 内两列由 RMode 偏移（0/2）选择。
//   双 subview 重载已由 TileOP #253/#254 修复，但 gfrun/gfsim 对双 subview
//   形态在运行期拒绝（#989 两层：校验断言 + 操作数绑定错位）；RMode 偏移
//   解码亦未修（#979）——当前用 pair 物化形态（V4.1），精度验证依赖 gfrun
//   本地补丁。
//   y 仍逐块窄存，(32,WIDE) 合并输出等 #968 修复后启用。
//
// ── InT 分支语义 ──
//   · bf16：数据与 recip 同为 bf16 → TROWEXPANDMUL 直乘（零转换）。
//   · half/fp32：recip 是 bf16、数据是 InT → dtype 不匹配。统一转到 fp32 域
//     物化后乘：TCVT(norm_f32, infrag) + TCVT(rf_f32, pair)，与静态版一致。
//
// ── 结构约束 ──
//   · 命名函数、裸指针参数：TileArray 在 lambda 内创建会地址逃逸（S32 spill）；
//   · noinline：B.IOT PE mask=1111 使 B.ASSEMBLE session 的 publish 需全部
//     4 个 PE 在同一 INIT pc 完成登记；内联成多份拷贝会让不同 PE 组执行不同
//     pc，session 永不 publish（gfrun FAIL E 实证）。单函数体 = 单 pc。
template <typename InT, int TileMv, int ValidRows, int WIDE, int BS,
          uint16_t RECIP_EMAX>
__attribute__((noinline))
void process_tile(InT *x, __fp8_e4m3 *y, uint8_t *scale,
                  int64_t M, int64_t N, int64_t scaleCols,
                  int64_t row0, int64_t kb0) {
    using namespace pto;
    using TT = TileTypes<InT, TileMv, ValidRows, WIDE, BS>;
    constexpr int BLKS = TT::BLKS;

    using gm_x = global_tensor<InT, RowMajor<-1, -1>>;
    using gm_y = global_tensor<__fp8_e4m3, RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    gm_x gx(x + row0 * N + kb0 * BS, static_cast<int>(M), static_cast<int>(N));
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
        TCVT(amax_arr[0][blk], mw);   // 读结果 + InT→fp32 加宽 + 写第 blk 槽
    }
    typename TT::t_f32_amax amax8 = TASSEMBLY<typename TT::t_f32_amax>(std::move(amax_arr));

    typename TT::t_bf16_wide mb, sh, ni, nz, ns, k, rp;
    auto amax_u32 = reinterpret_tile<uint32_t>(amax8);
    TANDS(amax_u32, amax_u32, FP32_EXP_MASK);  // 就地 fp32 域 floor（= golden 截断语义）。必须先
    //   floor 后转换：先 TCVT(RNE) 会把 amax ∈ [2^k-2^(k-9), 2^k) 向上舍过 binade，
    //   half/fp32 的 scale 差 2 倍（模型 TCVT float→float 默认 RNE，见 CubeEngine
    //   resolveTcvTRoundingMode；golden f32_to_bf16_bits 为截断）。bf16 值 fp32
    //   低位恒 0，此 floor 与原 bf16 域 TANDS 同果。
    //   ⚠ 本形态是 spec 预期写法（B.ASSEMBLE:38 / TANDS:40）：gfrun 逐字节 PASS，
    //   但 gfsim 的 #291 严格校验拒绝「组装产物作元素级源/目的」而 abort——已提
    //   issue 请求裁决统一（gfsim 侧，待编号）。过渡规避：TASSEMBLY 结果先 TCVT
    //   拷到普通 tile 再就地 floor（+1 op/tile，git 历史存档）。
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
    gm_s gsi(reinterpret_cast<__fp8_e8m0 *>(scale) + row0 * scaleCols + kb0,
             static_cast<int>(M), static_cast<int>(scaleCols));
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
        gm_y gyb0(y + row0 * N + (kb0 + 2 * p) * BS,
                  static_cast<int>(M), static_cast<int>(N));
        TSTORE(gyb0, oq0);
        gm_y gyb1(y + row0 * N + (kb0 + 2 * p + 1) * BS,
                  static_cast<int>(M), static_cast<int>(N));
        TSTORE(gyb1, oq1);
    }
}

} // namespace tail_ocp_fp8_dyn_detail

// ===========================================================================
// TAIL-OCP-FP8 运行期动态 shape 版（V4.2：TileN=256 + 拼装宽链 + pair 物化）
//   入口签名与 V1-V3 dyn 一致：tiling[0]=M、tiling[1]=N 运行期传入。
//   kernel 体镜像静态版 dynamic_mx_quant_tail_ocp_fp8（V4.2），差异仅：
//   M/N/scaleCols 走运行期参数 + RowMajor<-1,-1> 描述符。
//
// 【前提（本版不再防护，违反者静默算错或挂死）】
//   · M % 32 == 0：B.ASSEMBLE 跨 PE session 要求单一 process_tile 调用点
//     （noinline 单 pc）且行 tile 恒整 32 行——部分行需要第二个实例化点，
//     各 PE pc 错位 → session 永不 publish（挂死）；
//   · N % 256 == 0：TASSEMBLY parent 几何固定 (32,8)、TileArray 需满 8 个
//     fragment 槽——尾宽 tile 块数 <8 时尾部块被静默丢弃（无运行期防护）；
//   · 4-PE 全活跃且 TASSEMBLY 执行次数相等：行/列切分必须让每个 PE 的
//     myTiles×myKb > 0 且各 PE 相等（bench_small [64,16384]：2×2 各 32 次 ✓；
//     rowTiles=3（peRows=3、PE3 闲）或 numTiles<peCols 等形状会挂 session）。
//   放宽方向：尾行/尾列 tile 回退 V3 窄链路径（不用 TASSEMBLY）；oddTail
//   补齐列逻辑已在 N%256 前提下删除（numKb 恒 8 的倍数恒偶，scaleCols==numKb；
//   放宽 N 约束时需按 V3 的 ownsLastKb 模式恢复）。
//   单 PE（kPeNum=1）不支持——session mask=1111；请用 V1/V3 存档头。
// ===========================================================================
template <int BlockSize = 32, int kPeNum = 4, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8_dyn(InT *x, __fp8_e4m3 *y, uint8_t *scale,
                                       const int64_t *tiling) {
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of 32");
    static_assert(kPeNum == 4,
                  "V4.2 TASSEMBLY structure requires 4-PE sessions (B.IOT "
                  "mask=1111); for single PE use the V1/V3 archive headers");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>();
    constexpr int TileM = 32;                       // M32 cell 行高（固定）
    constexpr int WIDE  = 256;                      // TileN=256（V4.2）
    constexpr int BlocksPerTile = WIDE / BlockSize; // 8

    const int64_t M = tiling[0];
    const int64_t N = tiling[1];
    const int64_t numKb     = N / BlockSize;
    const int64_t scaleCols = ((numKb + 1) / 2) * 2;  // N%256 前提下 == numKb

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;

    // ---- L1 2D 切分（宽 tile 粒度）：行 tile 铺 peRows，余下 PE 切 N 轴 ----
    const int64_t itid = static_cast<int64_t>(tid);
    const int64_t rowTiles = (M + TileM - 1) / TileM;
    const int64_t peRows   = rowTiles < kPeNum ? rowTiles : kPeNum;
    const int64_t peCols   = kPeNum / peRows;        // 1→1×4、2→2×2、>=4→4×1
    const int64_t rowLane  = itid / peCols;
    const int64_t colLane  = itid % peCols;
    if (rowLane >= peRows) return;                   // 空 PE（session 前提见头注释）

    const int64_t tilesBase = rowTiles / peRows;
    const int64_t tilesRem  = rowTiles % peRows;
    const int64_t myTiles   = tilesBase + (rowLane < tilesRem ? 1 : 0);
    const int64_t myTileBegin =
        rowLane * tilesBase + (rowLane < tilesRem ? rowLane : tilesRem);
    if (myTiles <= 0) return;

    const int64_t numTiles = numKb / BlocksPerTile;
    const int64_t kbBase = numTiles / peCols;
    const int64_t kbRem  = numTiles % peCols;
    const int64_t myKb   = kbBase + (colLane < kbRem ? 1 : 0);
    if (myKb <= 0) return;                           // 行有份但列区间空
    const int64_t kbBegin = colLane * kbBase + (colLane < kbRem ? colLane : kbRem);

    for (int64_t t = 0; t < myTiles; ++t) {
        for (int64_t kb = kbBegin; kb < kbBegin + myKb; ++kb) {
            tail_ocp_fp8_dyn_detail::process_tile<
                InT, TileM, TileM, WIDE, BlockSize, RECIP_EMAX>(
                x, y, scale, M, N, scaleCols, (myTileBegin + t) * TileM,
                kb * BlocksPerTile);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
