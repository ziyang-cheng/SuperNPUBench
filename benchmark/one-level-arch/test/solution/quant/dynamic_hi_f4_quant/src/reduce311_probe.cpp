#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// #311 归约消费模式探针(新栈:model ed5b1d9e 含 PR#677 / TileOP f223dc5 含 TREDUCEPREFIXVIEW)
// ============================================================================
// 验证 fa_lowp.hpp 的官方模式能否解 hif4 归约输出消费(原 SSM #685 死结):
//   TROWMAX(宽 CUBE 载体) → TREDUCEPREFIXVIEW<Row>(取 prefix CELL) → TADD(dst,0,view) materialize
// 两个 [32,4] 组各出 [32,1] 组 max,再 TMAX 合并 → 验证下游二元 op 能正常消费。
//   make TESTCASE=dynamic_hi_f4_quant TYPE=REDUCE311_PROBE diss
//   gfrun -f <...>_reduce311_probe.elf   (期望不再 ValidateBasicBinaryTepl 失配)
// ============================================================================
using namespace pto;

static __bf16 x[32 * 8] __attribute__((aligned(4096))) = {};
static __bf16 y[32 * 1] __attribute__((aligned(4096))) = {};

int main() {
    using Src  = VecTileM32<__bf16, 32, 4, 32, 4>;   // [32,4] 组源(CUBE_M32)
    using Wide = VecTileM32<__bf16, 32, 4, 32, 1>;   // 宽归约载体(物理4 valid1)
    using Row  = VecTileM32<__bf16, 32, 1, 32, 1>;   // 紧凑 [32,1]

    global_tensor<__bf16, RowMajor<32, 4>> g0(x), g1(x + 32 * 4);
    global_tensor<__bf16, RowMajor<32, 1>> gy(y);

    Src s0, s1; TLOAD_CUBE(s0, g0); TLOAD_CUBE(s1, g1);
    Src a0, a1; TABS(a0, s0); TABS(a1, s1);

    Row zero; TEXPANDS(zero, 0.0f);

    Wide w0; TROWMAX(w0, a0);
    auto v0 = TREDUCEPREFIXVIEW<Row>(w0);
    Row m0;  TADD(m0, zero, v0);                       // materialize 组0 max

    Wide w1; TROWMAX(w1, a1);
    auto v1 = TREDUCEPREFIXVIEW<Row>(w1);
    Row m1;  TADD(m1, zero, v1);                       // materialize 组1 max

    Row vmax; TMAX(vmax, m0, m1);                      // 下游二元 op 消费紧凑 Row
    TSTORE_CUBE(gy, vmax);
    return 0;
}
