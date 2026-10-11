# ora/temporal-editor-test-v1

临时前半段诊断版本，2026-10-12。来源为云端 temporal_editor_v1 的通用证据 Prompt。
旧 Prompt 保留；版本中没有参考答案、视频 ID、场景专用规则或保留比例目标。

| 文件 | 用途 |
|---|---|
| discover.txt | 发现可能有剪辑价值的时段 |
| global.txt | 跨窗口复核与重复关系计划 |
| observe.txt | 不传候选描述的独立视觉观察 |
| core.txt | 纠正事件描述，再决定核心取舍 |
| boundary.txt | 起止点附近视觉复核 |
| verify.txt | 最后检查可见证据与价值 |
| duplicate.txt | 两个片段的实际视觉去重 |

JSON Schema 在 `pipelines/ora_temporal_editor_test/prompts.py`，实验冻结记录同时保存该源码和七份文本指纹。
已生成结果的目录不能修改 Prompt 后强行续跑。后续变更创建新 Prompt 版本、配置和新输出目录。
