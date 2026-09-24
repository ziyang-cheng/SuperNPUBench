# SuperNPUBench 联调 —「解决方案侧」Issue 状态汇总（持续累积）

> 由 `issue-tracking` skill 维护。判定以线上 GitHub 状态为准（issue state/reason + 关联 PR 是否合入 main），线上评论次之，聊天口头进展辅助。
> 累积规则：新 issue 追加；未关闭的重复 issue 刷新状态+日期；已关闭的冻结不再刷新。
> 最近更新：**2026-09-24**　｜　当前收录：**13** 个 issue　｜　来源：用户599305（王宇）SuperNPUBench联调群（9.21–9.22）
> 「提出人」= GitHub issue 作者（客观可核）；本批均由王宇在群内转达/催办。

---

## 状态总览表

| # | issue | 仓 | 提出人 | 当前状态 | 依据 | 最后核实 |
|---|---|---|---|---|---|---|
| 1 | [#195](https://github.com/LinxISA/Linx-TileOP-API/issues/195) | API | CCxx00 | ✅ **已完成** | closed/completed（09-21） | 2026-09-24 |
| 2 | [#762](https://github.com/LinxISA/SuperScalarModel/issues/762) | model | wangyuascend-spec | ✅ **已完成** | closed/completed；PR#764 合入 main（09-21） | 2026-09-24 |
| 3 | [#737](https://github.com/LinxISA/SuperScalarModel/issues/737) | model | Simona787 | 🔄 **部分修复（进行中）** | open；PR#792、#819 合入 main（修 layer1-2 + 层A）；第三层修复完成但待架构决策（PR 转 draft） | 2026-09-24 |
| 4 | [#765](https://github.com/LinxISA/SuperScalarModel/issues/765) | model | Cell-Cell | ✅ **已完成** | closed/completed；PR#815 合入 main（09-23） | 2026-09-24 |
| 5 | [#433](https://github.com/LinxISA/SuperScalarModel/issues/433) | model | Cell-Cell | 🔄 **修复已合入 main（issue 未关）** | open；PR#820 合入 main（09-23，正文点名修 #433） | 2026-09-24 |
| 6 | [#793](https://github.com/LinxISA/SuperScalarModel/issues/793) | model | ziyang-cheng | ❌ **未解决** | open；无合入 main 的修复 PR（我方提的 gfsim spin-wait） | 2026-09-24 |
| 7 | [#205](https://github.com/LinxISA/Linx-TileOP-API/issues/205) | API | CCxx00 | ❌ **未解决** | open；无合入 main 的修复 PR | 2026-09-24 |
| 8 | [#800](https://github.com/LinxISA/SuperScalarModel/issues/800) | model | CCxx00 | ✅ **已完成** | closed/completed；PR#799 合入 main（09-22，同 PR 亦修 #796） | 2026-09-24 |
| 9 | [#681](https://github.com/LinxISA/SuperScalarModel/issues/681) | model | CYR-Firework | 🔄 **有已验证修复（issue 未关）** | open；PR#783 合入 main（09-22）；四层家族根因已定位+验证（QSMLA 全部用例跑通） | 2026-09-24 |
| 10 | [#680](https://github.com/LinxISA/SuperScalarModel/issues/680) | model | CYR-Firework | ❌ **未解决** | open；无合入 main 的修复 PR | 2026-09-24 |
| 11 | [#665](https://github.com/LinxISA/SuperScalarModel/issues/665) | model | CYR-Firework | ❌ **未解决** | open；无合入 main 的修复 PR | 2026-09-24 |
| 12 | [#804](https://github.com/LinxISA/SuperScalarModel/issues/804) | model | wangyuascend-spec | ❌ **未解决** | open；无合入 main 的修复 PR（阻塞多个 normalization 用例） | 2026-09-24 |
| 13 | [#813](https://github.com/LinxISA/SuperScalarModel/issues/813) | model | CCxx00 | 🔴 **closed（原建议未采纳）** | closed/not_planned（09-22）；官方按 pto-spec 复核不采纳"非选中 PE 静默跳过"；底层行为由替代 PR#827（fail-closed）处理，PR#827 open 待合入 | 2026-09-24 |

## 状态统计

| 类别 | 数量 | issue |
|---|---|---|
| ✅ 已完成（closed + completed） | **4** | #195、#762、#765、#800 |
| 🔄 修复已进 main / 部分修复（issue 未关） | **3** | #433、#681、#737 |
| ❌ 未解决（open，无合入修复） | **5** | #793、#205、#680、#665、#804 |
| 🔴 closed（原建议未采纳） | **1** | #813 |

---

## 逐条说明

### #195（卷积 TIMG2COL/weight TLOAD BUG）— ✅ 已完成
09-21 即以 completed 关闭。

### #762（TCOLEXPAND CubeM32 RMS Norm broadcast）— ✅ 已完成
PR#764 已于 09-21 合入 main，issue completed 关闭。

### #737（gfsim GMMA cooperative CUBE 时序未建模，Cube Tileop=0）— 🔄 部分修复（进行中）
责任人（Simona787/flaviomarix）定位为三层根因 + 独立第四项（层A）：PR#792 修 layer1-2、PR#819 修层A，均合入 main；第三层修复完成但**待架构决策**（PR 转 draft），故 issue 仍 open。阻塞 quant_batch_matmul。

### #765（BSTART.TLSU 被劈开乱译）— ✅ 已完成
PR#815 已于 09-23 合入 main，issue completed 关闭。含从 #433 拆分出的"前端指令中间起始建块"根因。

### #433（SMT4 4-PE SPMD barrier 跨 PE 自旋标志不可见）— 🔄 修复已合入 main（issue 未关）
PR#820 已于 09-23 合入 main，正文点名"issue #433 定位的'已提交 store 被丢失'根因"，是该 issue 的修复。但 issue 本身仍 open（追踪状态未同步）。根因经勘误：从"跨 PE 陈旧读"→"已提交 store 被丢失"，部分拆到 #765。

### #793（gfsim C.BSTART.STD COND spin-wait 挂死）— ❌ 未解决
我方（ziyang-cheng）提的，已补复现评论；仍 open，无合入 main 的修复 PR。

### #205（movr 无法编码 S 寄存器目标）— ❌ 未解决
仍 open，无合入 main 的修复 PR。

### #800（movr 写 zero 寄存器目的端未建模）— ✅ 已完成
PR#799 合入 main（09-22），completed 关闭。同一 PR#799 亦修 #796（C.MOVR RegDst=0 discard，MXFP4 MatMul）。

### #681（GMMA 锁步身份 / SPMD group 家族）— 🔄 有已验证修复（issue 未关）
PR#783 已合入 main（09-22，用 SPMD group-op 游标替代 BROB slot 作锁步身份）。责任人（CYR-Firework）09-22 称"#681 家族四层根因已全部定位并有已验证修复——QSMLA 全部用例 + webcase 大 case 完整跑通"，09-23 补 3 个 QSMLA 复现用例。但 issue 仍 open（家族多层，或待全部合入/验证后关闭）。

### #680 / #665 — ❌ 未解决
均 open，无合入 main 的修复 PR。

### #804（gfsim CELLREG normalization Tile 寄存器不足）— ❌ 未解决
仍 open，无合入 main 的修复 PR，阻塞多个 normalization 用例，优先级较高。

### #813（单例 Shared 发布在非选中 PE 直接断言）— 🔴 closed（原建议未采纳）
09-22 关闭。官方（jiale-wangOwO）按 `pto-spec/main@01445483` 复核后不采纳其"非选中 PE 静默跳过"的建议——weight TLOAD / Shared TIMG2COL 是专用 bundle schema，不继承普通 Shared TLOAD 的 sparse-mask 行为。原 issue 诉求不成立；底层行为由替代 PR#827（`[gfrun][TLSU] fail closed special Shared masks`，fail-closed 方向）处理，PR#827 open 待合入。

---

## 关联发现（供参考）

- **#433 / #681 / #737 是同一类"gfsim 4-PE SPMD 跨 PE 可见性 / 锁步 / 死锁"问题族**，与 **#793** 同源（跨 PE volatile 自旋标志在 gfsim 时序模型不可见）。
- **#681 家族直接关联 QSMLA**：责任人用 QSMLA 全部用例作复现/验证载体。与我方实测的 QSMLA 4 个稀疏模式 gfsim `Deadlock detected` abort 属同一族；SWA（无稀疏）可跑通。#681/#433 家族修复全部合入并重编 model 后，值得实测确认 QSMLA 稀疏模式是否转好。
