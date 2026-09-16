#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// 最小复现:ROW reduction 输出「物理宽列」在 API(#137)与 model 之间不一致
// ============================================================================
// 背景:Linx-TileOP-API #137 "relax ROW reduction physical output shape" 放宽了
// TROWMAX 等的目的地约束——目的地物理列数 Cols 不再要求为 1,`R x 1` 仅表示归约结果的
// **有效区域**(valid col == 1)。pto-spec 也如此定义:
//   asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl:
//     "The destination has ValidRow equal to source.ValidRow and logical ValidCol
//      equal to one; its physical geometry is derived from the selected layout and
//      capacity."
//   asl/tile/model/legality/reduction-and-expansion.asl(行归约目的合法性):
//     destination_tile.valid_columns == 1        // 只约束 valid col,不约束 physical col
//
// 现象:把 TROWMAX 的目的地声明成物理 [32,2] + valid [32,1](#137 后编译通过),
// 再让一个普通二元 TEPL(TMAX)消费这个归约结果 → gfrun 运行期断言失败:
//   ASSERTION FAILED: IsCompatibleOperationDataTile(...) &&
//     "binary TEPL source dtype/shape/stride is incompatible"
//     (emulator/engine/AccumulateBlockInfo.cpp:982, func ValidateBasicBinaryTepl)
//
// 根因:model 把 TROWMAX 的**输出描述符**派生成 physical col==1(而非保留声明的
//   physical [32,2]),于是下游 TMAX 从 [32,2] 声明得到 physicalCol=2,却读到归约输出的
//   col=1 → IsCompatibleOperationDataTile(source->col(1) == physicalCol(2)) 为假。
//   即 model 落后于 #137/规范:归约输出的 physical col 仍被钉成 1。
//
// 期望:model 与 #137/规范一致——归约输出的 physical col 由声明容量派生(此处 2),
//   valid col == 1;下游按 physical col 消费,TMAX 通过。
//
// 编译(应 EXIT=0):
//   make TESTCASE=dynamic_hi_f4_quant TYPE=ROWREDUCE_WIDECOL_REPRO diss
// 运行(复现断言):
//   bin/gfrun -f <...>_rowreduce_widecol_repro.elf
// ============================================================================
using namespace pto;

static __bf16 x[32 * 4] __attribute__((aligned(4096))) = {};
static __bf16 y[32 * 2] __attribute__((aligned(4096))) = {};

int main() {
    using Src = Tile<Location::Vec, __bf16, 32, 4, BLayout::RowMajor>;          // [32,4]
    using Red = Tile<Location::Vec, __bf16, 32, 2, BLayout::RowMajor, 32, 1>;   // 物理[32,2] valid[32,1]

    global_tensor<__bf16, RowMajor<32, 4>> gx(x);
    global_tensor<__bf16, RowMajor<32, 2>> gy(y);

    Src xt; TLOAD(xt, gx);
    Red m;  TROWMAX(m, xt);   // 行归约:dst 物理[32,2]、valid[32,1](#137 后合法,编过)
    Red o;  TMAX(o, m, m);    // 普通二元 TEPL 消费归约结果 → gfrun 在此失配(model 输出 col=1)
    TSTORE(gy, o);
    return 0;
}
