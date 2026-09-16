# [SuperScalarModel/gfrun] 行归约输出被钉成 physical col == 1,与 Linx-TileOP-API #137 及 pto-spec 不一致

> **状态**:已提交 **SuperScalarModel issue #685**(https://github.com/LinxISA/SuperScalarModel/issues/685),等待上游修复。
> 在最新 main(`4bb0b2b4`)+ 最新工具链(llvm `46601e70` / TileOP `205cea8` 含 #137)上**仍复现**。

## 概述

Linx-TileOP-API **PR #137「relax ROW reduction physical output shape」**(commit `98426b1`,`linx` 分支)已放宽 `TROWMAX`/`TROWMIN`/`TROWPROD`/`TROWARGMAX`/`TROWARGMIN` 的目的地约束:**目的地物理列数 `Cols` 不再要求为 1**,`R × 1` 只描述归约结果的**有效区域**(`ValidCol == 1`)。

但 SuperScalarModel(gfrun)**未同步**:它仍要求行归约的 ROW_MAJOR 目的地 physical `col == 1`,并把归约**输出描述符**的 physical col 派生成 1。因此,当归约目的地按 #137 合法地声明成「物理 `[32,2]` + 有效 `[32,1]`」并被任一普通二元 TEPL(如 `TMAX`)消费时,gfrun 运行期断言失败。

**净效果:同一条内核在 API/编译期合法(#137),却无法在 gfrun 上运行——API 与 model 对「行归约输出物理列」的定义不一致。**

## 复现环境

| 组件 | 版本 |
|---|---|
| Linx-TileOP-API | `linx` 分支,含 **#137**(`98426b1`) |
| SuperScalarModel | `main`(复现于 tip `4bb0b2b4`,2026-09-15;更早版本同样存在) |

## 最小复现代码(自包含)

```cpp
#include <common/pto_tileop.hpp>
using namespace pto;

static __bf16 x[32 * 4] __attribute__((aligned(4096))) = {};
static __bf16 y[32 * 2] __attribute__((aligned(4096))) = {};

int main() {
    // 行归约目的地:物理 Cols = 2,ValidCol = 1。
    // 依 Linx-TileOP-API #137,这是合法声明(物理 Cols 不再被钉成 1)。
    using Src = Tile<Location::Vec, __bf16, 32, 4, BLayout::RowMajor>;          // [32,4]
    using Red = Tile<Location::Vec, __bf16, 32, 2, BLayout::RowMajor, 32, 1>;   // 物理[32,2] 有效[32,1]

    global_tensor<__bf16, RowMajor<32, 4>> gx(x);
    global_tensor<__bf16, RowMajor<32, 2>> gy(y);

    Src xt; TLOAD(xt, gx);
    Red m;  TROWMAX(m, xt);   // 行归约写入 物理[32,2] / 有效[32,1] 目的地
    Red o;  TMAX(o, m, m);    // 普通二元 TEPL 消费归约结果
    TSTORE(gy, o);
    return 0;
}
```

构建(linx 工具链,含 #137 的 TileOP 头):
```
clang++ -mlxbc -fenable-matrix -std=c++20 -D__linx -DENABLE_TENSOR_INSTR \
        repro.cpp <startup>.s -nostartfiles -o repro.elf
```
运行:
```
gfrun -f repro.elf
```

## 期望行为

`TROWMAX` 把 `[32,4]` 归约成有效 `[32,1]`,写入物理 `[32,2]`(有效列 1)的目的地;下游 `TMAX` 消费该 `[32,2]` tile 正常执行。依 #137 与 pto-spec:归约输出的 physical col 由声明的 layout/capacity 派生(此处 2),仅有效列为 1。

## 实际行为

`TROWMAX` 通过,但 `TMAX` 在运行期断言失败:
```
gfrun: illegal instruction: ASSERTION FAILED:
  IsCompatibleOperationDataTile(source, block->dataType, validRow, validCol, physicalCol, dstBytes)
  && "binary TEPL source dtype/shape/stride is incompatible"
, func ValidateBasicBinaryTepl, file emulator/engine/AccumulateBlockInfo.cpp
```

失配的实际字段(在断言处打印源与期望描述符得到):
```
op = TMAX
EXPECT : validRow=32  validCol=1  physicalCol=2  (来自 TMAX 目的地声明 [32,2])
SOURCE : validRow=32  validCol=1  col=1  layout=ROW_MAJOR  (即 TROWMAX 的输出)
```
`IsCompatibleOperationDataTile` 里 `source->tileInfo->col == physicalCol` 即 `1 == 2`,为假 → 断言失败。

## 根因

行归约的**输出描述符**被 model 固定为 physical `col == 1`(未保留目的地声明的物理 `[32,2]`)。于是下游二元 TEPL 从其目的地声明得到 `physicalCol = 2`,却读到归约输出源的 `col = 1`,二者不符。

涉及代码:
- `isa/Block.cpp` — 行归约目的地合法性(断言文案 `"PTO v0.58 row reduction requires final compatible source and destination descriptors"`)中的子条件:
  ```cpp
  (destinationCubeM || destination->tileInfo->col == 1)   // 对 ROW_MAJOR dst 仍强制 physical col == 1
  ```
- `emulator/engine/AccumulateBlockInfo.cpp` — `ValidateBasicBinaryTepl` 调用的 `IsCompatibleOperationDataTile`:
  ```cpp
  source->tileInfo->col == physicalCol   // 归约输出源 col=1,而下游 physicalCol=2
  ```
- 行归约输出描述符的派生把 physical col 定为 1(应改为由声明容量派生,保持有效列 1)。

## 规范依据(pto-spec)

- `asl/tile/reduce-and-expand/row-reduction/TROWMAX.asl`(legality):
  > "The destination has ValidRow equal to source.ValidRow and **logical ValidCol equal to one; its physical geometry is derived from the selected layout and capacity.**"
- `asl/tile/model/legality/reduction-and-expansion.asl`(行归约目的合法性):
  ```asl
  destination_tile.valid_rows == source_tile.valid_rows &&
  destination_tile.valid_columns == 1        // 只约束有效列;无 physical col 约束
  ```

两处均表明:归约目的地**只**要求 `valid_columns == 1`,physical col 由 layout/capacity 派生、不约束为 1。Linx-TileOP-API #137 已据此放宽 API;model 侧应同步。

## 建议修复

1. `isa/Block.cpp` 行归约目的地合法性:移除对 ROW_MAJOR dst 的 `col == 1` 强制,仅校验 `validCol == 1`(允许 physical col ≥ valid col、由容量派生)。
2. 行归约**输出描述符**派生:physical col 取自声明目的 tile 的容量/layout(如 `[32,2]` → col=2),而非钉 1;保持 `validCol == 1`。
3. 下游一元/二元 TEPL 的 `IsCompatibleOperationDataTile` 相应按 physical col 一致校验(声明侧与实际侧同为派生值)。

修复后,「物理列 > 1」的行归约输出可被下游正常消费,gfrun 行为与 #137 / pto-spec 一致。

## 影响

阻塞任何依赖「行归约输出物理列 > 1」的内核在 gfrun 上运行——典型场景:为满足 TCVT 的 physical-row 契约(narrowing TCVT 要求源/目的 capacity-derived 物理行相等)而把归约结果或其派生量声明成物理 `[32,2]` 的写法,会先合法编译(#137)、再在下游二元 TEPL 处被 model 拒绝。
