# [Linx-TileOP-API #63] column-reduction 的 B.DIM 错用 destination 几何 → 只归约 source row0（静默错算）

> 目标仓库：**LinxISA/Linx-TileOP-API** —— 官方已开 **#63**（column remainder，OPEN）
> 关联：SuperScalarModel **#560**（模型保持 source-geometry 解释）；PR **#69** 已修 row remainder
> 类型：regression（**静默** miscompile —— 编译通过、gfrun `R2=0`，数值错）
> 本文档为 dmxq 端到端补充证据（多行 source 逐元素 golden + 逐位 bisect + 行/列不对称机制）。

## 组件版本清单

| 组件 | 版本 | 说明 |
|---|---|---|
| **Linx-TileOP-API** | **`b8669ce`**（分支 `linx`）| 触发缺陷 |
| Linx-TileOP-API（对照好） | `804eb03` / `8677a6f`(=`ac8dcc5^`) | 编出正确结果 |
| llvm-project | `553b08045`（`dev-llvm15_56`）| 两组同一 clang |
| SuperScalarModel（gfrun） | `07e9c661`(+本地 TCMPS/ppoll) | 模型按 pto-spec source-geometry 校验 |
| SuperNPUBench | tag `ops-20260908` = `a3fa598` | dmxq 复现源 |
| pto-spec | `dea0b75e`（main）| `reduction-schema.asl` |

## 摘要

非尾轴 dmxq 沿 Axis（行）做 `TCOLMAX` 列规约，输出 `1×Post` 行向量。TileOP `b8669ce` 把
column-reduction 的 B.DIM 绑到 **destination** 几何（`1×N` → `LB1(ValidRow)=1`），使模型只归约
source 的 `row0` → scale 严重错 → 下游 output 饱和。**编译通过、gfrun 跑到底 `R2=0`，但数值错**。

## 复现（dmxq 端到端，多行 source 逐元素 golden）

```bash
export COMPILER_DIR=<...>/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<...>/linx_blockisa_llvm_musl/sysroot/usr
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant

# golden（多行 source：Axis=512 行，Post=256 列，BS=32）
python3 src/gen_dynamic_mx_quant_data.py --M 512 --K 256 --block-size 32 \
    --algo CUBLAS --kernel nontail --dtype FP8 --in-dtype fp16 --scale-layout compact --seed 42 \
    -o ../../../../compare/dynamic_mx_quant_nontail_cublas_fp8_4pe

make TESTCASE=dynamic_mx_quant TYPE=NONTAIL_CUBLAS_FP8_4PE res_check=on diss
GFRUN_FORCE_DIRECTBOOT_ABI=1 gfrun -f <elf> -s softcore.multiThreadNum=4
python3 src/dynamic_mx_quant_data_compare.py -d <elf> --dtype FP8 --scale-layout compact
```

结果对照（同 gfrun、同 golden、仅换 TileOP 头）：

| TileOP | output | scale |
|---|---|---|
| `804eb03`（好） | **pass MSE=0 MaxAE=0.0117** | **pass MSE=0** |
| `b8669ce`（坏） | **fail MSE=48492 MaxAE=447** | fail MSE=7.1 |

受影响的 4 个非尾轴用例（均 fail）：`nontail_cublas_fp8_4pe`、`nontail_cublas_fp8_bs128`、
`nontail_ocp_fp4_4pe`、`nontail_ocp_fp4_bs128`。3 个尾轴（`TROWMAX`）全 pass。

## 逐位 bisect（元凶 commit）

| commit | nontail_cublas_fp8_4pe | 说明 |
|---|---|---|
| `8677a6f`（=`ac8dcc5^`） | **pass MSE=0** | reduce B.DIM 仍 source 几何 |
| **`ac8dcc5`** | **fail MSE=48492** | 「bind expand/reduce B.DIM to **destination** geometry per ASL」|
| `67d47ab`(#82) | fail MSE=48492 | per-dim SS/SD/DS/DD sweep，未修 column |
| `b8669ce`（tip） | fail MSE=48492 | — |

**元凶 = `ac8dcc5`**（父提交 pass、本提交 fail，逐位定位）。

## 行/列不对称机制（为何尾轴幸存）

- `ac8dcc5` 把**行 TROW\* + 列 TCOL\*** 都绑 destination 几何（源码含 137 处 TCOL 命中）。
- 后续 `d3f8e47`(PR#69)「row-reduction B.DIM describes the **source** tile geometry」**只把 6 个行规约 TROW\* 改回 source**（row 104 命中 / col 20 且非修复）。
- 故 `b8669ce` 上：**行规约=source（对）→ 尾轴 TROWMAX pass**；**列规约=destination（错）→ 非尾轴 TCOLMAX 只读 row0 → fail**。

## 规范依据（pto-spec `dea0b75e`）

`asl/block/model/dispatch/reduction-schema.asl` `SelectedBundleClosedReductionSchemaLegal`
最终执行 `SelectedBundleComparisonShapeMatches(source)` —— row/column 用**同一** source-geometry 合同；
destination 的 `M×1` / `1×N` 由 source 单独派生，不应反向覆盖 B.DIM 的 source 几何。

## 最小修复方向（对称 PR#69）

对 `TCOLSUM/MAX/MIN/PROD/ARGMAX/ARGMIN` 六 op 的 SS/SD/DS/DD 四分支：
- 静态分支用 `tile_shape_in::{ValidCol,ValidRow,Cols}`；
- 动态分支用 `src.GetValidCol()/GetValidRow()` + source physical columns；
- destination 保持 `1×N` 独立 descriptor/capacity。

## 验收

行/列的 SUM/MAX/MIN/PROD/ARGMIN/ARGMAX 覆盖静/动态 source；反汇编 LB0/LB1/LB2 == source 几何；
column reduction 不再只算 row0；同一 ELF gfrun/gfsim 逐元素比独立 golden（本 dmxq 4 非尾轴 → pass）。
