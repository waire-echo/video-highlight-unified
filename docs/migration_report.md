# 统一重构与迁移报告

检查日期：2026-10-04，Asia/Hong_Kong。独立建立新目录，旧Git仓库、工作区修改、分支和远端保留。随后按用户要求实际归位本地资料、数据与结果，并核对哈希后清理重复文件和缓存；未删除旧仓库、强推或执行远端Archive。当前位置见 [本地整理说明](workspace_cleanup.md)。

## 检查与迁移

| 来源 | 核对与处理 |
|---|---|
| 本机/thekfjie `video-highlight-pipeline`，8f25180 | 工作区干净，作为新仓库起点，保留Git历史；迁移ORA/Spatial/接力/部署/预览/测试 |
| 本机 `video-highlight-editing`，78065e4 | 核对所有分支，保留独立main历史；存在未提交修改，有价值当前源码迁移并标hash与来源缺口 |
| waire-echo/video-highlight-editing远端，65fa242 | backup连接只读核对完整树，恢复9B revision/运行摘要，4B与9B core只差VERSION |
| thekfjie/video-highlight-editing远端，08a5d27 | 核对完整树，共同baseline/PGL/Qwen/ORA4B资产，没有发现独立reranker实现 |
| 独立TempSamp-R1，3fd4e85 | 官方center/qwen源码与Apache许可，训练框架保留上游链接 |
| handoff_balanced_42_20260816 | profile、来源commit/overlay、结果hash；重复源码去重，不整包搬迁 |
| cloud_temporal_20260816 / temporal_reaggregation_20260817 | raw原分数集中到artifacts/historical，登记统计/hash，实际验证balanced时间复现 |
| pgl_sum_temporal_20260819 / E:/AI/sources/MR.HiSum | PGL源码/测试/配置/学术非商业许可，保留负面结果与模型依赖 |
| results/ora4b_v5_new426_20261001 | 配置、revision、结果hash与统计；1095文件完整交接包集中到artifacts/deliveries/ora4b_v5_426/temporal |
| tmp/ora_stage2_new426_20261001 / 最终空间结果 | 主实现已在pipeline起点；保留交付hash/统计，临时部署/缓存不搬 |
| tmp/ora9b_v2_* | contracts/光流tracking参考与题面参考指标；缺temporal/pipeline/media_base，非完整可运行路线 |
| previews/ora_prompt_comparison_20260924/prompts | proposal Prompt/schema和v5请求样例；样例不等于运行模板 |
| 微信归档semantic_proposals_partial、ORA_candidates_raw_and_final、新版35空间包、ORA_RUN_v2 | 检查ZIP条目与请求证据，登记必要hash，原回答/预测留原位置 |
| submit/、exp07 ZIP、Qwen命名结果、根目录方案文档 | 成绩证据、失败结论、生成缺口汇总到history；不提交逐视频测试结果/个人背景 |
| TempSamp-R1/docs的56.43报告、E:/AI/tmp的论文及上传残留 | 原报告/论文集中到references/local；保留缺引用的Git对象资料；56.43报告参数独立登记 |

逐文件来源与原始SHA256：`docs/migration/source_manifest.json`。起点保留代码来自8f25180，Git能追踪文件移动。历史实验从registry进入详细来源记录。

## 哪些内容不迁移

保留源码、公共几何/IO/校验、Qwen/PGL/REZE、baseline、版本Prompt、关键配置、模型来源、指标摘要与测试。失败但有解释价值的实现也留下。

大型预测、回答、交接ZIP与预览已集中到被Git忽略的artifacts/；初赛视频集中到datasets/local/，权重和环境保留外部位置。通过artifact hash和更新后的本机路径索引定位。它们不等于垃圾，也不属于Git源码。

不把多份TempSamp-R1/verl副本、重复baseline展开、SSH/分块传输helper和私人Agent记录并入维护中的源码。一份有来源价值的旧源码/部署快照保存在本地artifacts/source_archives；经完整hash核对的重复交接解压、重复服务器下载、重复论文和可重建缓存已删除。独立训练框架保留来源链接，不并入当前比赛推理代码。

没有完整复制旧仓库到legacy。未找齐的结果不当作GT，不写逐视频特例规则。

## 当前路线

可执行：center/motion baseline、ORA4B/9B共用时间流程、Qwen direct/window、REZE-inspired、PGL-SUM、YOLO/Pose/BoT-SORT空间及预览。

参考资产：ORA proposal、ORA+Text Reranker、旧9B光流partial。SAM没有发现实现；复杂editing没有发现独立算法。可执行不代表本轮已验证所有GPU推理或语义准确率。

## 验证

工程测试、外置Prompt、配置分发、登记/续跑保护与CPU baseline本机验证。balanced对174条全量时间结果做真实重聚合对照。详细结果见 `docs/migration/validation.json`。

本轮没有连接服务器、运行新GPU模型推理、重评榜单或提交官网。重构不等于所有历史成绩都可复现。

## Archive建议

- `thekfjie/video-highlight-pipeline`：新入口验证后可Archive，README指向新仓库；其独立Git历史已在新仓库父历史保留。
- `thekfjie/video-highlight-editing`：可作为历史Archive，但先确认无未见分支/队友独立成果。
- `waire-echo/video-highlight-editing`：暂缓Archive；团队仓库，56.43生成链路、reranker和队友资产未全部交接，需共同确认。
- 独立TempSamp-R1：作为上游参考保留，不删除或改动官方远端。
- 本机旧工作区：保留，尤其未提交修改与独立main。本轮没有实际Archive/Delete。

## 仍无法确认

1. 56.43原作者、时间模型/Prompt和原始commit；新找到报告补出了空间参数，但报告48895帧与exp07 ZIP的45234帧不同，两个版本的对应关系仍未确认。
2. 46.55的榜单证据和其他submit产物的来源。
3. Reranker完整代码、revision、实际Prompt、候选/logits/预算日志。
4. proposal生成/整理脚本，只有code_sha256和请求证据。
5. ORA9B最终空间/榜单；9B v2只找到partial。
6. Qwen候选窗口v1-v7独立脚本/结果、合并前worker精确配置。
7. 本地Qwen确切revision、PGL特征环境freeze与32分最终空间产物。
8. SAM/复杂editing实现、三个队友尚未提供的其他仓库。

这些资产没有判成垃圾。补齐后沿用统一机制迁入，保留来源，不新建长期算法分支。
