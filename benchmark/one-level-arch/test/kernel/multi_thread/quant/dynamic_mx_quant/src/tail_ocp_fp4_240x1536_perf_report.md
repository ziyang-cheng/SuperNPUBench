# tail_ocp_fp4 性能报告 —— bf16 in, [240,1536], BlockSize=32

生成日期:2026-09-10 · 仿真器:SuperScalarModel `gfsim` · 工具链:linx_blockisa_llvm_musl(codex 基线,ops-20260908)

## 1. 被测对象与配置

| 项 | 值 |
|---|---|
| kernel | `dynamic_mx_quant_tail_ocp_fp4`(固定 SPMD 4-PE) |
| 输入 | `__bf16` [M=240, N=1536] |
| 量化 | OCP,**尾轴**(连续列)abs-max,BlockSize=32 |
| 输出 | `__fp4_e2m1x2`(2 元素/字节打包)[240, 768B/行] + scale(compact uint8 E8M0)[240, 48] |
| 倒数 | NEWCALC 位补(非 TRECIP,无 TCMPS) |
| ELF | `dynamic_mx_quant_tail_ocp_fp4.elf`(PM=240 PN=1536) |

`[240,1536]` 由 `[15360,1536]` 沿 M 轴缩小 64 倍得到;240/4=60(4-PE 整除)、1536/32=48(bs 整除)均合法。

## 2. 口径说明(重要)

- **精度**用 4 线程 `gfrun`(功能模型)验证,四个 PE 各处理 M 的 1/4(60 行),落盘完整 [240,1536] 输出再与金标逐字节比对。
- **性能**用 `gfsim`(周期级时序模型)。gfsim 对该 ELF `auto-detect: single-PE`,即使传 `multiThreadNum=4` 也**只模拟 tid=0**,处理其 1/4 切片(60 行×1536)。因此下文 `Total Cycles` 是**单 PE 时延**;4-PE 芯片上四片并发跑各自的 1/4,故该单 PE 时延即处理全量 [240,1536] 的**芯片墙钟**。
- 性能取数用**非 res_check** 构建(固定输入 x=4.0,不做文件 I/O);res_check ELF 在 gfsim 会因文件系统调用 segfault。

## 3. 精度结果 —— byte-exact 全过

4 线程 gfrun(`GFRUN_FORCE_DIRECTBOOT_ABI=1 … -s softcore.multiThreadNum=4`),各线程 3725 块、合计 14900 块、`R2=0`:

| 输出项 | 状态 | MSE | MaxAE |
|---|---|---|---|
| output(fp4 e2m1 打包) | **pass** | 0.000000 | 0.000000 |
| scale(compact uint8 E8M0) | **pass** | 0.000000 | 0.000000 |

金标由 `gen_dynamic_mx_quant_data.py --algo OCP --kernel tail --dtype FP4 --in-dtype bf16 --scale-layout compact --M 240 --K 1536 --block-size 32 --seed 42` 生成;比对 `dynamic_mx_quant_data_compare.py --dtype FP4 --scale-layout compact`。

## 4. 性能结果 —— 单 PE 18,069 cycles

### 4.1 顶层周期

| 指标 | 值 | 占 Total |
|---|---|---|
| **Total Cycles(单 PE ≈ 芯片墙钟)** | **18,069** | 100% |
| Vector Tileop Total Cycles | 11,476 | 63.5% |
| TLSU Tileop Total Cycles | 8,080 | 44.7% |
| Cube Tileop Total Cycles | 0 | — |
| Vector-TLSU union active | 17,979 | 99.5% |
| IPC(UOP) | 0.573 | — |

- Vector 与 TLSU 占用之和(11,476+8,080=19,556)略高于 union(17,979),说明二者**并发重叠仅约 1,577 cyc(≈8.7%)**,以串联流水为主;但 union 达 99.5% → **几乎没有"双执行单元同时空闲"的周期**,执行槽利用率很高。
- TLSU 拆分(PE-summed busy):Tload 9,793 / Tstore 17,842 —— 打包 fp4 写侧(NORM packed 行 stride)比读 bf16 更重。

### 4.2 Top-Down(Unified)

| 桶 | 占比 | 主要来源 |
|---|---|---|
| Retiring | 14.35% | ALU 13.78% + LD 0.56% |
| Bad Speculation | 12.98% | **Machine Clear 12.97%**(收尾 flush) |
| Frontend Bound | 0.28% | — |
| CMD Bound | 6.04% | BIQ capacity full 5.96% |
| **Backend Bound** | **66.35%** | BROB Stall 6.72% + 后端排队 |

尽管执行单元 99.5% 有活,块级 Retiring 只有 14.35%、Backend Bound 66% —— 典型 tile 机器画像:少量长延迟 tileop(TCVT / TROWMAX / TROWEXPANDMUL scale 广播 / 打包 TSTORE)占满执行槽,但等待依赖/队列容量,uop 级 IPC 低。

### 4.3 块级并行度与访存

| 指标 | 值 |
|---|---|
| Average Outstanding Block | 8.16(Vector 5.98 / TLoad 0.35 / TStore 1.83) |
| CellReg ReadNum / ReadLat | 9,914 / 5.29 cyc·op⁻¹ |
| CellReg WriteNum / WriteLat | 11,712 / 2.01 cyc·op⁻¹ |
| CellReg Bank Conflict | 7.50% |
| RowExpand Tile 数(scale 广播乘) | 100 |

## 5. 与 [15360,1536] 对比

| 指标 | [240,1536] | [15360,1536] | 比例 |
|---|---|---|---|
| Total Cycles(单 PE) | 18,069 | 1,024,344 | 1 : 56.7 |
| 理想线性(×64) | 16,005 | — | — |
| Vector Tileop | 11,476 | 634,958 | — |
| TLSU Tileop | 8,080 | 481,412 | — |
| IPC(UOP) | 0.573 | 0.630 | — |
| Retiring / BadSpec / Backend | 14.4% / 13.0% / 66.4% | 14.5% / 4.5% / 71.2% | — |

**小 shape 特性:**
- 实测 18,069 ≈ 理想线性 16,005 的 **1.13 倍**——固定的启动/收尾开销在小 shape 里占比放大。
- **Bad Speculation 从 4.5% 升到 13.0%**:收尾 Machine Clear(flush)是几乎恒定的一次性成本,在 1.8 万周期里权重远大于百万周期里。这是小 shape 单 PE 时延偏离线性的主因。
- IPC(UOP)略降(0.630→0.573),稳态窗口短、流水未充分铺满。
- 定性画像不变:**Cube 空转、Vector 主导(63%)、后端受限、访存与计算高利用率串联**。

## 6. 泳道图

同目录 `tail_ocp_fp4_240x1536.json`(Perfetto trace,25 MB)。在 <https://ui.perfetto.dev> 打开:
- TMA(TLSU)load/store slice 落在 `TLSU_<machineId>` 进程组(不在 CoreTop 主轨),观察 bf16 load 与打包 fp4 store 的 outstanding。
- Vector 轨观察 TCVT/TROWMAX/TROWEXPANDMUL 的连续占用与依赖链。
- 关注收尾段的 flush(对应 Top-Down 的 Machine Clear 12.97%)。

## 7. 复现命令

```bash
export COMPILER_DIR=<...>/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch
DMXQ=test/kernel/multi_thread/quant/dynamic_mx_quant
CHK=compare/dynamic_mx_quant_tail_ocp_fp4

# 金标
python3 $DMXQ/src/gen_dynamic_mx_quant_data.py --algo OCP --kernel tail --dtype FP4 \
  --in-dtype bf16 --scale-layout compact --M 240 --K 1536 --block-size 32 --seed 42 -o $CHK

# 精度(res_check + 4-PE gfrun),改宏必先清 tail_ocp_fp4 的 .o
find output -name '*tail_ocp_fp4*' -delete
make -C $DMXQ TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 res_check=on PM=240 PN=1536
GFRUN_FORCE_DIRECTBOOT_ABI=1 bin/gfrun -f <elf> -s softcore.multiThreadNum=4
python3 $DMXQ/src/dynamic_mx_quant_data_compare.py -d <elf> --dtype FP4 \
  --scale-layout compact --cmp-root $PWD/compare

# 性能(非 res_check + gfsim),同样先清 .o
find output -name '*tail_ocp_fp4*' -delete
make -C $DMXQ TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 PM=240 PN=1536
GFRUN_FORCE_DIRECTBOOT_ABI=1 bin/gfsim -f <elf>                                  # 周期
GFRUN_FORCE_DIRECTBOOT_ABI=1 bin/gfsim -f <elf> --swimlane 1 --swimfile <path>   # 泳道图
```
