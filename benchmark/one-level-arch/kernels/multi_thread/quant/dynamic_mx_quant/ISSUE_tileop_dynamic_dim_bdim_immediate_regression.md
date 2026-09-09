# [Linx-TileOP-API] 动态 tile 维度被误 lower 成立即数形式 → 所有动态 shape kernel 编译失败（回归）

> 目标仓库：**LinxISA/Linx-TileOP-API** —— 已提交为 **#100**
> 类型：regression（编译期）
> 严重度：高 —— 发布 tag `ops-20260908` 的工具链无法编译**其自身发布树内官方合入的**动态 shape kernel。

## 组件版本清单

| 组件 | 仓库 | 版本 | 说明 |
|---|---|---|---|
| **Linx-TileOP-API** | LinxISA/Linx-TileOP-API | **`b8669ce925a9`**（分支 `linx`）| **触发缺陷** |
| Linx-TileOP-API（对照） | 同上 | `804eb035b403` | 同代码编译**通过** |
| llvm-project | LinxISA/llvm-project | `553b08045111`（分支 `dev-llvm15_56`，clang 15.0.4）| 两组测试同一 clang |
| SuperNPUBench | PTO-ISA/SuperNPUBench | tag `ops-20260908` = `a3fa59899dc4` | 复现所用官方 kernel 源 |
| pto-spec（权威规范） | PTO-ISA/pto-spec | `dea0b75e803c`（main）| 契约依据 |

> 说明：llvm/clang、SuperNPUBench 源、复现命令在两组测试中**完全一致**，唯一变量是 Linx-TileOP-API 头文件版本（`b8669ce` vs `804eb03`）。

## 摘要

带**运行期（动态）维度**的 tile —— 即 `Tile<..., ValidRow=-1, ...>` / `global_tensor<T, RowMajor<-1,-1>>` —— 在 `b8669ce` 上编译失败：

```
tileop-api/jcore/template_asm.hpp:2379:5: error: invalid operand for inline asm constraint 'i'
tileop-api/jcore/template_asm.hpp:2599:5: error: invalid operand for inline asm constraint 'i'
tileop-api/jcore/template_asm.hpp:9911:6: error: Match Instruction Error!
```

运行期维度被降级 lower 成**立即数 `B.DIM` 形式**（`"i"` 约束），而不是 pto-spec 规定的**寄存器源 `B.DIM RegSrc` 形式**（`"r"` 约束）。这不是某个 kernel 的写法问题：**发布树内 5 个官方合入的动态 kernel 全��同样失败**（见下）。

## 复现步骤（用已合入主线的官方代码，可直接复现）

前置：用 tag `ops-20260908` 的工具链（TileOP `b8669ce` + llvm `dev-llvm15_56@553b08045`）构建出 `linx_blockisa_llvm_musl`，导出：

```bash
export COMPILER_DIR=<output>/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<output>/linx_blockisa_llvm_musl/sysroot/usr
```

复现（**SuperNPUBench@ops-20260908 自带的官方 solution kernel，无需任何第三方改动**）：

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/solution/normalization/rms_norm
make TESTCASE=rms_norm COMPILER_DIR="$COMPILER_DIR" DType=__half G_A=512 G_R=8192 PE_NUM=4 diss
# => template_asm.hpp:2379 invalid operand for inline asm constraint 'i'
```

**同一回归影响的全部官方主线 kernel**（均 `rc=2`，同一 `constraint 'i'` 签名）：

| 官方 kernel（`kernels/solution/`）| 复现目录 |
|---|---|
| `normalization/rms_norm` | `test/solution/normalization/rms_norm` |
| `normalization/rms_norm_binary` | `test/solution/normalization/rms_norm_binary` |
| `normalization/group_norm_grad` | `test/solution/normalization/group_norm_grad` |
| `normalization/group_norm_grad_1d` | `test/solution/normalization/group_norm_grad_1d` |
| `quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8_dyn` | （上游合入版）|

每个目录均有 `compile.all` 一键复现。

## 预期 vs 实际

- **预期**：动态 kernel 编译通过（`804eb03` 上 `rms_norm` `rc=0` 编过；kernel 源一字未改）。
- **实际**：`b8669ce` 上编译失败，运行期维度撞 `"i"` 立即数约束。

## 根因

`template_asm.hpp` 的 TLOAD 路径（2379 起）对三个维度**硬编码为立即数形式**：

```asm
"B.DIM zero, %c[VCOL], ->lb0\n"     ; %c + "i"(valid_col)
"B.DIM zero, %c[VROW], ->lb1\n"     ; %c + "i"(valid_row)   <- valid_row 是运行期值 → 崩
"B.DIM zero, %c[COL],  ->lb2\n"
"B.IOT mask=1111, last, ->%[d0]<%Z[TileSize]>\n"
: ... [VCOL]"i"(valid_col), [VROW]"i"(valid_row), ...
```

当 tile 的 `ValidRow`（或 `ValidCol`）是运行期值时，`"i"` 约束无法满足。仓内**存在**动态（寄存器）形式（如 `template_asm.hpp:~9935`）：

```asm
"B.DIM %[src____dimcol], 0, ->lb0\n"   ; "r"(src.GetValidCol())  <- 正确的动态形式
```

即分派逻辑（`TLoadBackend.hpp:288` 的 `if constexpr (... ValidRow==DYNAMIC || ...)`）本应把运行期维度路由到寄存器形式，但实际仍落到立即数模板。附带 `B.IOT ... <%Z[TileSize]>` 报 `Match Instruction Error`。

## 权威依据（pto-spec）

pto-spec `dea0b75e` 明确动态维度是一等 ISA 特性，**没有**要求任何维度必须立即数：

- **ADR-BLOCK-0012 Decision 013**：`B.DIM.RegSrc` 码 `0..23` 命名 24 个绝对 GPR（含零寄存器）。
- **ADR-BLOCK-0012 Decision 014**：`B.DIM` 读所选 GPR + 17-bit 无符号立即数偏移，写入 LB。
  - 静态维度 = `RegSrc=zero(码0) + imm`（或压缩形式 `C.B.DIMI imm8`）；
  - **动态维度 = `RegSrc=装着运行期值的 GPR`**（`docs/block/attributes/B.DIM.md`）。
- **Decision 015**：每个 `LB0/LB1/LB2` write-once（`B.DIM` 与 `C.B.DIMI` 二选一）。

即 LB0/LB1/LB2 三个维度**都可动态**。工具链把运行期维度强行 lower 成立即数形式、无 GPR-source 路径，**违反上述契约**。

## 回归定位

- `804eb03` ✅ 编过；`b8669ce` ❌ 崩。区间 `804eb03..b8669ce` 内改动 B.DIM 维度 lowering 的嫌疑 commit：
  - **`67d47ab` (#82) TileOP: per-dimension static/dynamic B.DIM lowering (SS/SD/DS/DD)** — 2026-09-07（主嫌，重做静/动态维度分派）
  - **`c88ba58` (#92) TileOP: eliminate scalar stack round-trips and fix SD-branch dim binding** — 2026-09-08
  - 相关：`870495a` (#73)、`d3f8e47` (#69)、`ac8dcc5`

## 影响范围

发布 tag `ops-20260908` 工具链**无法编译其自身发布树内任何动态 shape kernel**（≥5 个官方 solution kernel）。任何依赖运行期 tile 维度的 kernel 均受阻。

## 建议

修复 TLOAD（及同款 range-subview 2599、`B.IOT` TileSize 9911）的维度 lowering：当某维度 `Valid* == DYNAMIC` 时，发 `B.DIM <gpr>, 0, ->lbN`（`"r"` 约束、RegSrc=运行期 GPR），而非 `B.DIM zero, %c[...]`（`"i"`）；仅静态维度用立即数/`C.B.DIMI`。以官方 `rms_norm`（`test/solution/normalization/rms_norm`）作回归门即可。
