# 前半段 ORA / GPT 参考对照试验

入口说明和完整判断在 [测试说明](../../docs/temporal_reference_test.md)。

输入为原视频、原 PTS 与已有数据来源清单；输出为候选、盲观察、取舍理由、边界复核、原帧号与错误记录。
只验证时间选段。默认同一批 12 条，无自动全量。GPT 参考由离线工具读取，推理程序没有参考答案参数。

- 代码来源：云端临时 `temporal_editor_v1`，2026-10-11 下载的原文快照。
- 新版本：`temporal-editor-test-v3`；[配置](../../configs/experiments/ora-temporal-editor-test-v3.json)。
- Prompt：[`ora/temporal-editor-test-v3`](../../prompts/ora/temporal-editor-test-v3/README.md)。
- 启动：`launch.py --data-root ... --video-dir ... --runtime-root ... --model-path ... --output ...`。
- CPU 验收：同上添加 `--check-only`；指定视频：`--ids 20,21,313`。
- 按原 PTS 导出 `temporal_handoff.jsonl`，此文件是试验对照接口，尚未注册到生产 workflow。
- 原型的空结果二次检查矛盾、候选超过 64、长目录超过预算仍会明确失败；其他视频继续。

运行目录的 `frozen.json` 记录实际 commit、Prompt/源码/配置 SHA256、权重来源、原视频与 PTS 来源。
`raw/` 保留实际请求、实际视频片段来源、处理器帧预算和回答；`records/` 保留每条的判断。
`errors/` 保留失败视频；当前失败列表以 `report.json` 为准。失败视频不会导出为正常空结果。
`unresolved` 与 `quality_flags` 必须一起检查；短片完整保留可能合理，flag 只是复核提醒。

模型仍为同一 ORA4B。增加盲观察会多一次候选复核调用，是本次诊断成本。
单视频帧预算 128，双视频去重仍限制每条 32；这些是上限，不保证处理器实际读取了全部原帧。
GPU 协议与新结果质量尚待验证。
