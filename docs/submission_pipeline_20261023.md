# 比赛提交全流程固化

日期：2026-10-23

入口：

```text
scripts/run_submission_pipeline.py
```

本入口把已经验证的组件串成一条只推理、不训练的提交流水线：

```text
Video-ORA-4B + checkpoint-118 LoRA
  -> 523 个窗口推理
  -> 严格合并
  -> 宽容修复
  -> 原帧号时间交接
  -> v7E Stage 2 主体跟踪和裁剪
  -> 426 行 predictions_spatial.jsonl
```

## 新权重

本目录用于第二轮 `checkpoint-228` 的新权重全量运行：

```text
/media/kubevolume/teacher1/video-highlight/runtime/
  sft426-doubao237-continue-adapter118-lr3e5-world3-v13-e1-20261023/
  trainer/checkpoint-228
```

Adapter SHA256 固定为：

```text
37b5d75def0c4c9061efc405495c3d5bcc6ace4b60c509657a68c0d90678c790
```

总控在推理前重新计算并校验该哈希，不能用同名路径误切到其他权重。
新运行目录为 `runtime/submission-pipeline-new-20261023`，不复用父 Adapter
的 523 个回答。

## 已有严格结果续跑

为节省已经完成的 523 窗口 ORA 推理，可显式导入完整严格 merge：

```bash
/path/to/python scripts/run_submission_pipeline.py \
  --seed-strict-merged /path/to/merged-snapshot \
  --start-stage merge \
  --execute
```

导入时要求严格快照为 426 个源视频、523 个窗口、509 个有效窗口
和 14 个失败窗口，并记录四份输入的 SHA256 与来源。

## 时间区间规范化

模型回答先解析原始区间，再按开始时间排序；重复、重叠和首尾相接的
区间直接合并，不连续的区间保持独立。合并完成后才检查字段、有限数值和
视频时间边界。

16 段不是提交格式限制，也不是当前 adapter 推理的硬失败条件。连续 1 秒
区间会先合并，例如 0 到 4 秒的四个相接区间会规范化为 `0-4`。合并后若
仍有 20 个真正不连续的区间，则保留 20 段；数量只写入运行审计，不触发
自动重跑，也不强行压缩到 16。模型把大量时间判为高光仍可能在语义上错误，
这属于高光选择质量，而不是区间解析错误。

## 运行绑定

每次运行生成独立 `run-<pipeline_id>`，不覆盖历史运行。pipeline ID
包含基础模型、Adapter、523 窗口 manifest、426 视频索引、推理参数、
宽容阈值、v7E 复现包、YOLO 权重、包装器和总控代码的 SHA256。
输入变化时保留旧运行并更新 `CURRENT.json` 指向新运行。

## 输出

最终目录：

```text
runtime/submission-pipeline-20261023/run-<pipeline_id>/06-submission/
```

主要文件：

```text
predictions_spatial.jsonl
submission.jsonl
submission.zip
FINAL.json
```

`submission.jsonl` 是同一字节内容的兼容别名；ZIP 只包含
`predictions_spatial.jsonl`。`FINAL.json` 记录行数、预测帧数、输入收据、
SHA256 和 `website_submission_performed=false`。

## 空高光视频

Stage 1 的 `valid=false` 且 `segments_frame=[]` 表示该源视频确实没有高光。
Stage 2 包装器直接生成 `predictions=[]`，并执行相同的比例和空预测校验，
不会再调用只接受非空片段的包内 `validate_stage1`。最终仍然输出 426 行。

## 安全边界

- 不训练、不更换 Adapter、不覆盖第一轮或第二轮训练产物。
- 不读取 v7E 复现包参考时间戳作为答案。
- 不因单个窗口失败而伪造空预测；宽容修复仍有未解决窗口时停止。
- 不自动向比赛官网提交。
