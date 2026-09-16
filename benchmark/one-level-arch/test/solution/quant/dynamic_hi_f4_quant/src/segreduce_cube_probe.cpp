#include <common/pto_tileop.hpp>
#include <cstdint>
#include <utility>

// ============================================================================
// CUBE_M32 分段-max 发射探针(V1 蓝本 de-risk,2026-09-15)
// ============================================================================
// 目的:验证 V1 伪码「相邻分段 max」在当前工具链(TileOP 987d034)的**可发射
// 通路**。旧的 TRESHAPE 窄列路线已被 segreduce_probe.cpp 证伪(RECORD 问题10);
// 新路线借用仓库里**已过 gfrun 的**两个惯用法:
//   - TPARTVIEW<Sub,1,Parts>(parent):把 Matrix-location CUBE tile 切成 Parts 个
//     列子视图(fa_subview.hpp fa_subview_row_max,过 gfrun)。
//   - TileArray + TASSEMBLY<Parent>(std::move(array)):把若干 sub-tile 组装回宽
//     tile(matmul_quantize.hpp assemble path,gfrun R2=0)。
//
// 本探针只验最陌生的一环:64 → 16 相邻-4 max。
//   parent [32,64] --TPARTVIEW 16×[32,4]--> 各 TROWMAX --> 16×[32,1] --TASSEMBLY--> [32,16]
//
// 约束依据(pto_tile.hpp / pto_tile_region.hpp):
//   - TPARTVIEW parent 必须 is_legal_subview_parent(Matrix location + CubeLayout)
//     → 用 CubeTileM32(Location::Left)作 parent,不能用 VecTileM32(Vec)。
//   - TROWMAX dst 必须物理 [N,1](ValidCol==1 && Cols==1)。
//   - TASSEMBLY 要求 parent/fragment 的 dtype/location/layout/shape/字节覆盖全匹配。
//
// 编译:make TESTCASE=dynamic_hi_f4_quant TYPE=SEGREDUCE_CUBE_PROBE diss
// 期望:EXIT=0 + diss 含真实 BSTART.TEPL(TABS/TROWMAX)+ 范围修饰(B.SUBVIEW/
//       B.ASSEMBLE)。若 TASSEMBLY 的字节覆盖 static_assert 失败,则暴露 CUBE_M32
//       [32,1] 的 LogicalTileBytes padding 与 [32,16] 不整除的问题,须调整方案。
// ============================================================================
using namespace pto;

static constexpr int M  = 32;   // CubeM32 fractal 行 = 32
static constexpr int BS = 64;   // hi_f4 固定 block

static __bf16 x    [M * BS] __attribute__((aligned(4096))) = {};
static __bf16 vm16 [M * 16] __attribute__((aligned(4096))) = {};

int main() {
    // 免压缩 MAX 树:TPARTVIEW 切 16 个 [32,4] 子视图 → 各 TROWMAX 出独立 [32,1] →
    //   相邻对 TMAX 锦标赛树 16→8→4→2→1。全程 TROWMAX+TMAX,无 subview 组装/压缩/TPACK。
    //   验证「分段 max 不需要密集压缩即可发射」。
    using ParentTile = CubeTileM32<__bf16, M, BS>;   // Left [32,64]
    using SubTile    = CubeTileM32<__bf16, M, 4>;    // [32,4] 相邻-4 子视图
    using MaxSlot    = CubeTileM32<__bf16, M, 1>;    // [32,1] 单组 rowmax

    global_tensor<__bf16, RowMajor<M, BS>> gX(x);
    global_tensor<__bf16, RowMajor<M, 1>>  gM(vm16);

    ParentTile xv;   TLOAD_CUBE(xv, gX);
    ParentTile vabs; TABS(vabs, xv);

    // 第一层:16 个相邻-4 子视图各出独立 [32,1]。region TROWMAX 收 SubTileView 具名左值。
    auto parts = TPARTVIEW<SubTile, 1, 16>(vabs);
    MaxSlot m16[16];
#pragma clang loop unroll(full)
    for (int p = 0; p < 16; ++p) {
        auto part = parts[0][p];
        TROWMAX(m16[p], part);
    }
    // 锦标赛树 16→8→4→2→1(相邻对 TMAX)。
    MaxSlot m8[8];
#pragma clang loop unroll(full)
    for (int p = 0; p < 8; ++p) TMAX(m8[p], m16[2 * p], m16[2 * p + 1]);
    MaxSlot m4[4];
#pragma clang loop unroll(full)
    for (int p = 0; p < 4; ++p) TMAX(m4[p], m8[2 * p], m8[2 * p + 1]);
    MaxSlot m2[2];
    TMAX(m2[0], m4[0], m4[1]);
    TMAX(m2[1], m4[2], m4[3]);
    MaxSlot vmax;
    TMAX(vmax, m2[0], m2[1]);

    // ---- 验证 scale/recip/predicate 位运算族在 CUBE_M32 [32,1] 可发射 ----
    // (数值不求正确,只验 op 类型/发射:TMULS/reinterpret_tile/TSHRS/TANDS/TSHLS/
    //  TADDS/TMUL/TCMPS<GE>/TEXPANDS/TSEL/TROWEXPANDMUL)
    auto bf16c = [](uint16_t b) { return __builtin_bit_cast(__bf16, b); };
    MaxSlot sf; TMULS(sf, vmax, bf16c(0x3E12));            // SF = vmax * (1/7)
    // e6m2→rec 位重构(占位式,只验位运算族发射):就地在 sf 的 uint16 视图上运算。
    auto sfu = reinterpret_tile<uint16_t>(sf);             // uint16 视图(lvalue,就地)
    TSHRS(sfu, sfu, (uint16_t)2);                          // >>2
    TANDS(sfu, sfu, (uint16_t)0x3F);                       // &0x3F(取 exp6)
    TSHLS(sfu, sfu, (uint16_t)7);                          // <<7
    TADDS(sfu, sfu, (uint16_t)0x3F80);                     // + LUT 基(占位)
    // 现 sf(plain bf16)持有 rec 位。L2:t8 = Vmax8*rec。
    MaxSlot t8; TMUL(t8, m8[0], sf);
    MaxSlot pred; TCMPS<pto::CmpMode::GE>(pred, t8, bf16c(0x4080)); // ≥4 谓词
    MaxSlot f8;   TEXPANDS(f8, bf16c(0x3F80));             // 1.0
    MaxSlot half; TEXPANDS(half, bf16c(0x3F00));           // 0.5
    TSEL(f8, pred, half);                                  // (≥4)?0.5:1.0
    MaxSlot vnorm; TROWEXPANDMUL(vnorm, vmax, f8);         // 广播乘(验发射)

    TSTORE_CUBE(gM, vnorm);
    return 0;
}
