# [gfsim][VECTOR] TROWEXPAND 广播源被限制为「单个 128B CELL」，与 pto-spec 冲突 → 广播源 >128B 的 kernel 时序仿真 abort

> 目标仓库：**LinxISA/SuperScalarModel**（gfsim 时序模型）—— 已提交 **#605**
> 类型：时序模型契约**过严**（over-constraint，非 kernel/工具链问题）；编译/功能(gfrun)正常，仅 gfsim abort
> 复现用**已合入 main 的 solution dmxq kernel + solution test**，不依赖任何私有分支。

## 组件版本清单

| 组件 | 版本 | 备注 |
|---|---|---|
| **SuperScalarModel（gfsim）** | `main` @ `07e9c661`（+ ppoll cherry-pick `5491fa6e`）| 断言在 `TimingSim/pe/vec/VecTop.cpp:1601`；ppoll 修复仅为**越过启动期**才够得到此断言 |
| SuperNPUBench | tag `ops-20260908` = `a3fa598` | `kernels/solution/quant/dynamic_mx_quant` + `test/solution/quant/dynamic_mx_quant` |
| llvm-project | `dev-llvm15_56` @ `553b08045` | clang 15.0.4 |
| Linx-TileOP-API | `b8669ce` | 与工具链版本无关 |
| pto-spec（规范依据） | `dea0b75e` | TROWEXPAND / reduction-and-expansion 契约 |

## 摘要

gfsim 时序模型对 `TROWEXPAND`（行广播）的广播源施加了**「必须放进一个 128B CELL」**的断言，而 pto-spec 对广播源**只约束形状（单列 + 行数等于目的行数）、无任何字节/CELL 尺寸上限**。任何广播源 >128B 的合法 kernel 在 gfsim 上 abort。gfrun（功能模型）与编译均正常，仅 gfsim 时序仿真崩。

```
[gfsim] caught std::exception: ASSERTION FAILED:
  broadcastBytes > 0U && broadcastBytes <= VEC_CELL_GRANULARITY &&
  "TROWEXPAND broadcast source must fit one 128B CELL"
, func ValidateRowExpandContract, file TimingSim/pe/vec/VecTop.cpp:1601
```

## 复现（已合入 main 的 solution dmxq kernel）

```bash
export COMPILER_DIR=<...>/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<...>/linx_blockisa_llvm_musl/sysroot/usr
cd SuperNPUBench/benchmark/one-level-arch/test/solution/quant/dynamic_mx_quant

make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP8 COMPILER_DIR="$COMPILER_DIR" diss
gfsim -f <...>/output/solution/quant/dynamic_mx_quant/elf/dynamic_mx_quant_tail_ocp_fp8.elf
# => VecTop.cpp:1601 ASSERTION FAILED: broadcastBytes ... must fit one 128B CELL
```

同样 abort 的 mainline solution 尾轴 kernel（均用 `TROWEXPANDMUL`）：
`TAIL_OCP_FP8`、`TAIL_OCP_FP4`、`TAIL_CUBLAS_FP8_4PE`（后者加 `-s softcore.multiThreadNum=4`）。

> 注：需 ppoll 修复（`5491fa6e`，已并入 main）才能越过 hosted-musl 启动探测、够到此时序断言；否则更早在启动期 abort。

## 具体触发（反汇编铁证）

`dynamic_mx_quant_tail_ocp_fp8`（BlockSize=32 → `TileM=64`）的数据 pass 末对 fp32 数据按每行 recip 广播相乘，反汇编：

```
BSTART.TEPL  TROWEXPANDMUL, FP32
C.B.DIMI  32, ->lb0
C.B.DIMI  64, ->lb1      ; 广播源有效行数 = TileM = 64
C.B.DIMI  32, ->lb2
```

gfsim `VecTop.cpp:1599-1603`：
```cpp
const uint64_t broadcastElements = cmd->lb1;              // = 64
const uint64_t broadcastBytes = broadcastElements * elementBytes; // 64 * 4(FP32) = 256
ASSERT(broadcastBytes > 0U && broadcastBytes <= VEC_CELL_GRANULARITY /*128B*/ &&
       "TROWEXPAND broadcast source must fit one 128B CELL");
```
广播源 `[64 行, 1 列]` fp32 = **256B**，跨 2 个 128B CELL → 断言失败。

## 规范依据（pto-spec `dea0b75e`）：广播源无尺寸上限

1. **`docs/tile/reduce-and-expand/row-expansion/TROWEXPAND.md`**：广播源为「one one-column vector」，逐位广播到每个有效行；物理列宽由 layout 派生 —— **无尺寸/CELL 限制**。
2. **`asl/tile/model/legality/reduction-and-expansion.asl:224-230`** 与 **`asl/block/model/dispatch/expansion-schema.asl:84-92`**：广播合法性**只查形状**——
   行扩展要 `broadcast.valid_columns == 1 && broadcast.valid_rows == destination.valid_rows`；
   **无任何字节数 / CELL 数上限**。`valid_rows` 可达整个 tile。
3. **`B.IOT` SizeCode 1..10 = 128B..64KB per PE**（`asl/block/operands/B.IOT.asl`）——单个 Local tile 合法尺寸远超 128B，故广播源 256B 完全合法。

即 gfsim 的「must fit one 128B CELL」是**时序模型自造的过严约束，规范里不存在**。

## 隔离（纯 gfsim，与工具链无关）

同一 gfsim、同一 kernel，仅换 TileOP 头：`804eb03` 与 `b8669ce` **均触发同一断言**（两版 TROWEXPANDMUL 都发 `lb1=64`）。故与 TileOP 的 B.DIM lowering（另见 Linx-TileOP-API #63/#100）**无关**，是纯 gfsim 时序建模缺口。gfrun 功能模型对同 ELF 正确跑通 `R2=0`。

## 修复方向

`ValidateRowExpandContract`（及列扩展对应处，若同样限制）应**按 `ceil(broadcastBytes / VEC_CELL_GRANULARITY)` 个 CELL 建模广播源读取时序**，而非断言广播源 ≤ 1 个 CELL。仅保留 `broadcastBytes > 0` 与形状（单列 / 行数匹配目的）校验，去掉 `≤ VEC_CELL_GRANULARITY` 上限。

## 验收

- solution `TAIL_OCP_FP8/TAIL_OCP_FP4/TAIL_CUBLAS_FP8_4PE` gfsim 不再 abort、跑到 Report Stop 并给出 cycles。
- 广播源跨 1/2/N 个 CELL 各有单测；与 gfrun 同一 ELF 的功能结果一致。
