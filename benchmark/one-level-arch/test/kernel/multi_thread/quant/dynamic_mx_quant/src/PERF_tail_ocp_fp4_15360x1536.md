# 性能分析报告 — `dynamic_mx_quant_tail_ocp_fp4` [M=15360, N=1536] BlockSize=32

> 方法遵循 `SuperScalarModel/modelSpec/performance_analysis_guide.md`：三个可观测面联用
> （**PMU Top-Down 分诊 → PMU 深层子块定位 → PipeView/SwimLane 可视化验证**），
> 逐层下钻 L1 → Backend →（Memory vs Core）→ 执行单元 → **Vector 内部 uop/CellReg 路径** → 下游依赖。
> 配套 artifact（同目录）：`tail_ocp_fp4_15360x1536.json`（SwimLane 稳态窗）、
> `tail_ocp_fp4_192x1536.json`（SwimLane 小 case 完整）、`tail_ocp_fp4_15360x1536_pipeview.out`（Konata PipeView）。

## 1. 用例与构建

| 项 | 值 |
|---|---|
| kernel | `dynamic_mx_quant_tail_ocp_fp4<15360, 1536, 32>`（尾轴量化，bf16 in → fp4(e2m1) 打包 out + e8m0 scale） |
| 输入/输出 | bf16 [15360,1536]=47.19MB → fp4 packed 11.80MB + e8m0 scale [15360,48]=0.74MB；numKb=48 |
| SPMD | 固定 4-PE，按 M 行切分（每 PE 3840 行）。4 个 PE-线程**并发于同一核、共享同一条 3,686,979 周期墙钟**（非四等分）；每 PE 独立 VEC ALU / TLSU / CellReg，SFU/CUBE StgBufB 为 4-PE 共享（本 kernel 不用） |
| 模拟 | `gfsim --conf fourpe --pto-v02 true`（按指南 §1.1 对 SPMD 规程真实建模 4 线程） |

### ⚠ TileM / gfsim #605（构建前提，勿删）
工作树 kernel 为 **TileM=128**（release TileOP #119 TCVT 契约），该构建 gfsim 撞 #605 abort。本报告用 **TileM=32**
构建规避 #605（广播源 128B=1 CELL），TileM 是 tiling 常量、不改输出（gfrun 逐字节相等）。周期是 TileM=32 tiling 下的。

---

## 2. 性能现状（Performance Status）——★ V1 基线

> **本报告 = V1 性能基线（baseline-v1）。** 后续任何优化都以此为对照，且须固定同一测量条件才可比：
> Bench tag `ops-20260908` · TileM=32 tiling（#605 规避）· `gfsim --conf fourpe --pto-v02 true` · 1.65GHz 换算 · seed 常量输入。
> **V1 锚点：3,686,979 cyc / ≈2.235 ms / 向量发射率 0.175 uop·cyc⁻¹·PE⁻¹（峰值 2 的 8.7%）**；优化后复测须同时保证 gfrun 逐字节不变。

单次量化整块 [15360,1536]（4-PE 并发，gfsim `--conf fourpe`，时钟按 CubeCore 标注 1.65 GHz）：

| 现状指标 | 数值 |
|---|---|
| **总周期（wall-clock）** | **3,686,979 cyc** |
| **墙钟时延 @1.65GHz** | **≈ 2.235 ms** |
| **元素吞吐** | **6.40 elem/cyc ≈ 10.56 Gelem/s**（每 PE 1.60 elem/cyc） |
| 访存吞吐 | 输入读 21.1 GB/s、输出写 5.6 GB/s、**聚合 26.7 GB/s（16.2 B/cyc）** |
| tileop 速率 | 460,800 tileop / 3.69M cyc = **0.125 tileop/cyc** |
| **向量发射率** | **0.175 uop/cyc/PE**（模型 `VECTOR Utilization`=issued/总周期，与 IPC(UOP) 同一量）= **archSpec 发射峰值 2 uop/cyc/PE 的 ≈8.7%** |
| IPC(UOP) | 0.175（=上行发射率同一量） |
| 数据规模 | in bf16 47.19MB → out fp4 11.80MB + scale 0.74MB（23.6M 元素，numKb=48） |

**一句话现状**：当前该 shape 需 **≈2.24 ms / 3.69M cyc**，元素吞吐 **10.6 Gelem/s**、聚合访存 **26.7 GB/s**；但向量单元
发射率仅 **0.175 uop/cyc = 发射带宽峰值(2/cyc)的 ≈8.7%**，发射远未打满——**瓶颈见 §4–§5（Vector 受 CellReg 源读延迟 +
SrcBuf 反压限制）**。（注：8.7% 是相对发射带宽峰值；严格的"可提速倍数"需 roofline 下界，本报告未作此估算。）

> 口径：以上为 TileM=32 tiling（gfsim 可跑）下的模型仿真值；TileM=128 本尊预计更快（§8.5）。时延按额定 1.65GHz 换算，
> 未含 kernel 外的启动/同步开销。

---

## 3. 第一层：Unified Top-Down（slot 闭合，sums=100%）

| L1 | 占比 | L2/L3 关键项 |
|---|---|---|
| Retiring | 16.06% | ALU 15.43% / LD 0.62% |
| Bad Speculation | 7.21% | **Machine Clear 7.20%**（Branch Mispred 0%） |
| Frontend Bound | 0.00% | — |
| CMD Bound | 0.00% | — |
| **Backend Bound** | **76.74%** | Memory Bound **0.00%** / **Core Bound 76.74%** |

Core Bound 按执行单元：**Vector 48.65%** / TLSU Tstore 12.45% / TLSU Tload 9.30% / Scalar ALU 6.34% / Cube 0 / Idle 0。

- 总 wall-clock **3,686,979 cyc**；`Run 3,677,129 + Idle 9,850 = Total` ✓，All Cores Idle 仅 0.27%。
- **Memory Bound = 0** → 数据全在片上、非访存瓶颈；瓶颈是**执行端口（Core Bound），且 Vector 一家独大**。

## 4. 第二层：Vector 端口内部——不是"算不过来"，是"喂不上"

只看 Top-Down 会停在"Vector 48.65%"。下钻 `Vector PMU Stats`（每 PE 一致）与 `TileOp Efficiency` 才见根因：

> 口径提醒：以下 `*Cycles` 是 plain 周期计数（可 /总周期 3,686,979 得墙钟占比）；`*UopCycles` 是 uop×周期聚合量
> （同 `Vector Busy` 的坑，**不能当墙钟%**，只能看总量或均摊到每 uop）。

| 指标（PE0，四 PE 相同；分母=总周期 3,686,979） | 值 | 含义 |
|---|---|---|
| **向量发射率** = issued/总周期 | **0.175 uop/cyc**（= IPC(UOP)） | = archSpec 发射峰值 2 uop/cyc 的 **8.7%** |
| TileopNum / UopNum | 115,200 / 645,120 | 每 tileop 拆 ~5.6 个 128B CELL-uop |
| **UopQDispatchSrcBufFullBlockedCycles** | **581,760 = 总周期的 15.8%** | SrcBuf 满**阻塞 uop 派发**的周期（plain cycle 计数，口径正确） |
| UopQDispatchSrcBufFullCycles | 610,560 = 16.6% | SrcBuf 处于满态的周期 |
| ReadLat / bucket | 5.07 cyc（LatLE5 597,938 / LatLE10 24,142） | 每次 CellReg 源读往返 ~5 拍 |
| WriteLat | 2.00 cyc | 写回快，不是写侧瓶颈 |
| RowExpandSourceResponseWaitUop**Cycles** | 760,320 uop-cycle（**聚合量,非墙钟%**） | 均摊 **~4.1 cyc/行广播uop**（760,320 / 184,320） |
| RowExpandUopSplit / Src0 读 / Broadcast 命中 | 184,320 / 184,320 / 178,560 | 广播源已 **96.9% 命中 buffer**；真正的源读是**数据操作数 src0 每 uop 一次（184,320 次）** |
| VecSrcCellNotReadyCycles | 0 | ALU 输入端本身不缺——是 CellReg 源读往返 + SrcBuf 占用限住发射 |

**定位**：向量端口是瓶颈（§3 的 48.65%）。机理有据的量：**发射率仅 0.175/cyc（峰值 2/cyc 的 8.7%）**、**SrcBuf 满
阻塞派发 15.8% 的周期**、**每次 CellReg 源读往返 ~5.07 拍**。这些源读来自 `TROWEXPANDMUL`（行广播乘 `xf * recip_f`）：
广播源 recip_f 已 96.9% 命中广播 buffer，真正的成本是**逐元素乘必须读一遍数据操作数 src0（184,320 次/PE、每次等 ~5 拍）**。
所以天花板是 **CellReg 源读延迟 + SrcBuf 反压限制了发射**，而非向量算术吞吐。

## 5. 第三层：下游 TLSU 是被向量拖住，非独立瓶颈

`TLSU tile-store STQ residency`（PE0）：resident 4,561,368 entry-cycles，其中 **98.99% = "tile data (rename) not ready"**，
`SCB full 0% / older-store 0% / 指针阻塞 0%`。即 store 排队等的是**向量单元尚未产出的数据**，store 通路本身无结构性阻塞。

→ Top-Down 里 TLSU Tstore 12.45% / Tload 9.30% 是 Vector 瓶颈的**下游后果**；`Vector-TLSU overlap 76%`、STQ 全在等
tile-data，都印证访存被藏在向量阴影里。

## 6. 佐证：OOO 窗口、块构成、误预测

| 指标 | 值 | 判读 |
|---|---|---|
| Average BROB depth | **751.65** | OOO 窗口很深——**不是 ILP/窗口不足**，是执行端口喂不动 |
| Outstanding Vector Block | 3.37（TStore 1.90 / TLoad 0.74） | 在飞块以向量为主 |
| Retired PARALLEL 块构成 | **VCALL(向量) 86.96%** / TLSU 13.04% / CUBE 0 | 工作量 87% 是向量块 |
| Machine Clear 7.20% | inter-block misp **642 次** | 少量跨块误预测被深 BROB 窗口放大成 7.2% slot 浪费 |
| CellReg Bank Conflict | 0.28%（WQ_VEC Lose 0% / WQ_TLSU Lose 2.73%） | bank 争用可忽略 |

## 7. PipeView / SwimLane（可视化验证）

- **PipeView** `tail_ocp_fp4_15360x1536_pipeview.out`（Konata，`-m 1200`，23,486 cyc）：VEC uop 走 `SB→RD→RSP(读)→RDY→EXEC→…→RSP(写)`。
  按指南 §2.4/§2.5，重点看 **`RSP`(读) 条宽 ≈ CellReg 往返(~5 拍) 且 `RDY` 条久**——即 uop 在等源、执行资源空闲，直接对应 §4 的
  源读延迟与 0.175 uop/cyc 低发射率。（trace 内 RSP 事件 63,360、RDY 31,104、EXEC 32,256。）
- **SwimLane** `tail_ocp_fp4_15360x1536.json`（4-PE 稳态窗 77,348 cyc）/ `tail_ocp_fp4_192x1536.json`（4-PE 小 case 完整
  61,846 cyc）：看 `VECTOR_*` 四线程轨长期铺满、`TLSU_*` 与其重叠、`CellReg conflict` 近乎空。

## 8. 结论与优化方向

1. **根因不是"Vector 算力吃满"，而是 Vector 发射受限于 CellReg 源读延迟 + SrcBuf 反压**：发射率仅 0.175 uop/cyc（峰值 2/cyc
   的 8.7%），源读往返 ~5 拍、SrcBuf 满**阻塞派发 15.8% 的周期**；这些源读主要是行广播乘 `TROWEXPANDMUL` 的数据操作数 src0
   （184,320 次/PE）。下游 TSTORE 的 STQ 98.99% 在等向量产出的 tile 数据。Memory Bound=0、Cube=0、前端/CMD 无气泡、OOO 窗口深（751）。
2. **优化优先级（按影响排序）**：
   - **① 减少向量 uop 总量**：行广播乘把每 tile 拆 32 uop（184,320/PE），每个 uop 读一次 src0 并等 ~5 拍——uop 数量直接决定
     源读次数。方向：减少每 tile 的 uop 拆分、或缩短产生这些 tile 的算子链。（广播源 recip_f 已 96.9% 命中 buffer，无进一步空间。）
   - **② 缩短向量算子链**：fp4 特有的 inf/zero/special **三守卫 TCMPS+TSEL×3** 是额外 uop 源，评估能否合并判定或减少中间 tile。
   - **③ 隐藏源读延迟**：加深 SrcBuf / 增大在飞源读，以掩盖 ~5 拍 CellReg 往返（当前 SrcBuf 满阻塞派发 15.8% 周期）。
3. **访存/bank 不用动**（Memory Bound 0、bank 冲突 0.28%、TLSU 端口是下游后果）。
4. **次要项**：Machine Clear 7.2% 源自 642 次跨块误预测被深窗口放大，可后续查该分支来源。
5. **待 #605 修复后复测 TileM=128 本尊**：更大 tile → 每 PE 循环更少、行广播 uop 总数下降，Vector 源读压力与总周期预计随之
   下降（功能/精度不变）——当前最有价值的一次复测。
