#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4.hpp"
using namespace supernpu::tile_isa::mxquant;

// Non-tail OCP-FP4, LARGE BlockSize (=128) 用例。曾由方案A 切归约轴的 `_bigbs` kernel
// 覆盖（现已退休）；新工具链 tile 上限 256KB 后，单块 [128,64] bf16=16KB 合法，统一 public
// 入口 dynamic_mx_quant_nontail_ocp_fp4<Axis,Post,BlockSize> 对大 BS 回退 TileN=align=64
// 走 plain 单块。实测 gfrun 逐字节 == 旧 bigbs 输出。
// Axis=128 (=BlockSize, numKb=1), Post=64 (TileN=64=2 MX blocks)。x=[128,64],
// y=[128,32] bytes。scale compact planar uint8 E8M0 [scaleRows, Post]:
// scaleRows = evenAlign(Axis/128) = 2 -> [2,64] bytes。
//
// SINGLE-PE: numKb=1 无块行并行度，故 kPeNum=1（tid 0 全包），无 tid gate / 屏障。
static __bf16 x[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_ocp_fp4<128, 64, 128, __fp4_e2m1x2, __bf16, /*kPeNum=*/1>(
        x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
