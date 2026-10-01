# 输入和输出字段

## 原始数据和索引

视频按数字ID命名，例如 `0.mp4`、`1.mp4`。配套原始配置是同名JSONL，允许仅包含 `targetRatioWH`；工具会从文件名补出ID。索引是JSON数组：

```json
[
  {"video_id":"0","targetRatioWH":[16,9]},
  {"video_id":"1","targetRatioWH":[9,16]}
]
```

ID不可重复，视频和配置须逐一匹配。画幅为 `[16,9]` 或 `[9,16]`。

## 时间交接

下面是字段示例，数字为教学用途：

```json
{
  "schema_version":"temporal-handoff/v1",
  "video_id":"0",
  "targetRatioWH":[16,9],
  "frame_count":200,
  "segments_frame":[[10,20],[40,50]],
  "segments_sec":[[0.333333,0.7],[1.333333,1.7]],
  "subject_hint":"person",
  "action_description":"Synthetic example",
  "valid":true,
  "errors":[]
}
```

- 帧号从0开始，属于原视频。
- `segments_frame` 的 `[a,b]` 包含两端，例如 `[10,12]` 对应10、11、12三帧。
- `segments_sec` 与实际选择的帧段配对，使用原视频PTS。
- `event_segments_sec` 另存模型连续时间区间，可能因帧量化与配对秒数略有差异。
- `highlight_frames.jsonl` 中的列表将上述帧段逐帧展开。
- 主体和动作说明只是后续模块的自动提示，不是比赛提交字段。

## 最终空间文件

每个视频一行。高光内必须稠密包含每一帧：

```jsonl
{"video_id":"0","targetRatioWH":[16,9],"predictions":[{"frame":10,"bboxes":[0,20,320]},{"frame":11,"bboxes":[0,20,320]}]}
{"video_id":"1","targetRatioWH":[9,16],"predictions":[]}
```

`bboxes` 只写 `[x,y,w]`，不写高度。高度为 `w*target_h/target_w`。框必须完全位于原画面，帧号严格递增且不重复。

`submission.zip` 只包含一个 `predictions_spatial.jsonl`。预览MP4和诊断报告留在本地运行目录。
