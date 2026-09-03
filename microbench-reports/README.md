# 微基准精度看护报告

由 `microbench-tracking` skill **手动**生成（区别于 `../reports/` 每日自动版本追踪、`../pr-reports/` 待合入 PR 验证）。

用 SuperNPUBench PR #96 的 `tileop-guard` harness（126 个 demo + host 独立 numpy golden）对 PTO-ISA 微基准做**精度看护**：不是看「能不能跑」，而是看「**算得对不对**」——四态判据（编译失败→执行失败→精度失败→精度正确），golden 钉 pto-spec 规范 ASL 的预期语义。精度失败=模型缺口 witness（已挂 issue），非我方回归。

子目录按**发布 tag** 归档（配套工具链/模型随 tag 发布）。最新在最上。

| 时间 | 基线 tag | 精度正确 | 精度失败(witness) | run-only | 编译失败 | run-fail | 报告 | 归档 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-03 21:33 +0800 | `ops-20260828` | 101 | 11 | 4 | 2 | 8 | [报告](ops-20260828/2026-09-03T2133+0800__microbench-guard.md) | [json](ops-20260828/2026-09-03T2133+0800__microbench-guard.json) |
