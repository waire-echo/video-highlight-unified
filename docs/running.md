# 运行说明

`vh.py` 输入配置、模型/媒体路径、数据索引，输出时间 handoff 或逐帧裁剪 JSONL。新实验复制配置，不复制整套脚本。

| 配置 | 用途 | 状态 |
|---|---|---|
| `baseline_center.json` | 全片居中，合成示例 | CPU可跑 |
| `baseline_motion.json` | 帧差时间+居中 | CPU可跑 |
| `ora4b_v5.json` | 默认ORA+空间 | GPU，原流程有完整交付 |
| `ora9b_v1.json` | 同一ORA代码，9B权重 | GPU对照 |
| `qwen_direct.json` | 直接秒数定位 | 对照，有片头偏置 |
| `qwen_window_v1.json` | 窗口评分/融合 | 对照，有评分饱和 |
| `reze_yesno_v1.json` | Yes/No logits + Otsu | 对照，非原论文完整复现 |
| `pgl_sum_v1.json` | PGL重要度+边界 | 对照，历史有片尾误选 |
| `spatial_crop.json` | handoff→检测/姿态/跟踪 | 保持时间帧集合 |
| `ora_proposal_v1.json` | 历史请求配置 | 参考，拒绝运行 |
| `ora_reranker_reference.json` | 缺失链路记录 | 参考，拒绝运行 |

`parameters` 放算法参数。`--local` JSON 放 `video_dir`、`test_index`、模型文件和解释器路径。默认 `--action temporal`，ORA `--action all` 加空间；`spatial/prepare/check/validate/status` 分阶段操作。

## ORA 调试

`pipeline.py` 保留低层分阶段接口。`configs/pipeline.example.json` 是它使用的扁平配置；`configs/pipelines/*.json` 是 `vh.py` 使用的统一实验配置，两种格式不可互换。

```bash
python pipeline.py --config configs/pipeline.local.json --dry-run all
python pipeline.py --config configs/pipeline.local.json temporal --plan
python pipeline.py --config configs/pipeline.local.json temporal --smoke 5
python pipeline.py --config configs/pipeline.local.json prepare --handoff /data/temporal_handoff.jsonl
python pipeline.py --config configs/pipeline.local.json spatial --handoff /data/temporal_handoff.jsonl --smoke-videos 0,1,2
python pipeline.py --config configs/pipeline.local.json validate
python pipeline.py --config configs/pipeline.local.json preview --random 3
```

低层入口适合调试。正式成绩实验使用 `vh.py` 登记。ORA 的窗口/FPS/上下文/padding 由配置传入，未知参数报错。空间 `auto/compat` 保留原档位参数，需要新阈值时扩展同一实现的配置接口。

空间参数可在统一配置增加 `spatial_parameters`，如 `{"yolo":{"conf":0.15,"stride":3},"pose":{"smooth_window":15},"tracking":{"min_score_margin":0.06}}`。未填写项沿用记录在源码commit中的原版默认值，输入/输出/权重路径受到保护。batch/concurrency由runtime档位决定。

## balanced 时间复现

有历史原始 handoff 和 raw 时，不需要模型/GPU/视频：

```bash
python scripts/reaggregate_temporal_handoff.py --handoff artifacts/historical/qwen_window_20260816/temporal_handoff.jsonl --raw artifacts/historical/qwen_window_20260816/temporal_handoff.jsonl.raw.jsonl --out-root runs/balanced-reproduction --profiles balanced
```

历史 profile 在 `experiments/records/hist-qwen-balanced-42/config.json`。这里只复现时间选择，不能声称重现42分完整流程。

## Editing

`tools/preview_submission.py` 支持原画框、裁剪、左右对比，`pipeline.py preview` 已集成。`scripts/preview_temporal_handoff.py` 查看时间。目前没有独立复杂镜头编排、配乐或转场算法。
