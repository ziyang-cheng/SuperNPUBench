# 微基准精度看护报告

由 `microbench-tracking` skill 每日自动生成（区别于 `../reports/` 版本可用性、`../pr-reports/` 待合入 PR）。
看 PTO-ISA 微基准**算得对不对**（四态 + host golden 精度）。最新在最上。

| 时间 | 基线 tag | 精度正确 | 精度失败(witness) | run-only | 编译失败 | run-fail | 报告 | 归档 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-03 21:33:00 | `ops-20260828` | 101 | 11 | 4 | 2 | 8 | [报告](ops-20260828/2026-09-03T2133+0800__microbench-guard.md) | [json](ops-20260828/2026-09-03T2133+0800__microbench-guard.json) |
