# 历史 YOLO 与 Pose 空间方案

状态：`reference-only`。依据 2026-07-15 原始报告恢复参数；未找齐运行代码、时间 hints、最终结果和 commit，不能作为可直接运行的完整 pipeline。

输入是队友提供的高光帧与主体提示，输出是逐帧 `[x,y,w]`。空间阶段使用 YOLO11m 检测和 15 帧轨迹平滑；16:9 人物垂直锚点受 0.45 倍裁剪高度限制；9:16 人物由 YOLO11m-Pose 肩髋横坐标修正，融合比例 0.50、最大位移 0.35 倍裁剪宽度。报告还描述 0.25 秒内部时间边界 trim。

参数原文转存于 [历史配置快照](../../../experiments/records/hist-yolo-pose-5643-report/config.json)。原始报告位于 `references/local/reports/yolo_56_43_20260715.docx`，通过 `tools/locate_artifacts.py --experiment hist-yolo-pose-5643-report` 验证哈希。

报告声称平台得分 56.43、最终 48,895 帧；已定位的 exp07 ZIP 为 45,234 帧。尚无证据证明两个文件版本一致。原报告还包含不适用于当前正式比赛限制的模型规模建议，应以赛题和当前规则为准。

当前 YOLO/Pose 公共实现见 `src/video_highlight/spatial/`，是否与此历史方案完全等价需要原始脚本对照。缺失的 `trim_prediction_segments.py`、`build_best_spatial_submission.py` 不通过猜测重建。
