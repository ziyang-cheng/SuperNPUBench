# SuperNPUBench 版本追踪 / 看护报告分支

本分支（`version-tracking-reports`）**只存放报告、不含源码**，按三个场景分成三个并列命名空间，各自维护索引，互不干扰。

| 命名空间 | 场景 | 看什么 | 判据 | 生成方式 | 索引 |
|---|---|---|---|---|---|
| [`reports/`](reports/README.md) | 已发布 release 版本的全量算子 | **能不能跑** | gfrun/gfsim PASS/FAIL | `version-tracking` skill（每日自动） | [reports/README.md](reports/README.md) |
| [`pr-reports/`](pr-reports/README.md) | 待合入 PR 的算子 | **能不能跑** | 编译 / gfrun / gfsim | `pr-tracking` skill（手动） | [pr-reports/README.md](pr-reports/README.md) |
| [`microbench-reports/`](microbench-reports/README.md) | 微基准 tile 算子 | **算得对不对** | 四态 + host golden 精度 | `microbench-tracking` skill（手动） | [microbench-reports/README.md](microbench-reports/README.md) |

各命名空间下均按发布 tag 分子目录归档（如 `<namespace>/ops-20260828/`），报告文件名以本地时间戳开头。

> 每日自动看护只写 `reports/`（含其 `README.md` / `index.tsv`）；`pr-reports/`、`microbench-reports/` 与本页为手动维护，互不影响。
