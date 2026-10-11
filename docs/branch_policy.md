# 主线与分支规则

## 唯一维护主线

`main` 是唯一维护主线，包含统一 workflow、TUI、WebUI、冻结兼容版和独立实验版本。当前整理对象为 scene_general_v3。代码经开发分支和 PR 合入 main；服务器维护工作区检出 main，不长期停在已完成的功能分支。

代码分支不是运行时算法版本。运行时版本在 `configs/workflows/versions/`，默认版本由 `configs/project_policy.json` 指定。三个既有清单不覆盖；算法继续测试时增加新源码目录和清单，不改默认兼容版。

基准算法提交为 `36695a2`；交接为 `81f2257`；原始结果基准标签为 `ora-spatial-v3-baseline-20261011`。这些表示历史算法和结果，不代表 main 必须永远停在这些提交。

## 每次开发的流程

1. 从最新 main 建立短期 feat / fix / exp 分支，明确问题和对应 version ID。
2. 修改编排、界面、I/O 或文档；core、冻结 v3 与已有 Prompt 不变。新算法写入独立目录。
3. 提交代码、版本清单、测试及功能说明；数据、模型、完整预测、机器配置和密钥不提交。
4. PR 描述触发条件、最终行为、验证结果及 GPU 待验收范围。
5. 自动检查通过、版本与默认配置无歧义后合并 main。尚未通过 GPU 验收的功能保留 pending 标记，不替换默认兼容版。
6. 服务器部署明确的 main 提交，检查入口和来源；确认代码可从 main 获取后删除已经合并的开发分支。

本次工程整理通过 PR #1 归入主线。主线可以包含明确标记 GPU pending 的实验功能；它不构成模型回答或全量推理等价性已经验证的结论。

## 自动化约束

每次 push / pull_request 自动执行 Engineering contracts：

- 安装声明的 dev 测试依赖和 CPU Torch；
- 编译 Python；
- 完整 pytest，包含 41 个冻结文件与版本 SHA256、项目默认版本/文档入口，以及原 v3 的 10 项测试；
- Git 文件审计，拒绝视频、预测、模型、机器配置和密钥进入提交。

独立冻结快照在 `configs/workflows/frozen_contracts.json`。运行 `python tools/check_scene_baseline.py` 与 `python tools/check_project_policy.py` 可重复检查。审查时同时检查约束脚本、快照、测试和 CI 本身，防止绕过检查。

运行时另有检查：源文件指纹、输入、权重、提交、完成阶段收据和产物哈希不一致时拒绝续跑。新实验使用新目录，不强行改旧 frozen.json 来续跑。

## GitHub 权限限制

2026-10-11 UTC 核实：本私有仓库套餐对 branch protection / rulesets 返回 HTTP 403，要求支持私有仓库保护的套餐。仓库保持私有，未更改套餐。因此目前 CI 可以报告违规，不能强制阻止有写权限的人直接推送 main 或忽略失败合并。

需要 GitHub 侧强制 PR 和必需状态检查时，应由仓库所有者启用相应套餐并配置规则。现有源码和运行时校验继续生效，不把它们称为强制分支保护。

## 发布与回滚

服务器查看真实部署，不在 README 写死容易过期的 main commit：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

每次发布使用带日期的工程版本标签。GPU 验收另行登记，不通过标签名称暗示已验收。回滚检出明确的旧提交；旧运行在原提交下续跑，新提交创建新运行。
