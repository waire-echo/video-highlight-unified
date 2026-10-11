# 分支、自动化与版本约束

2026-10-11 UTC 核实：main 仍为 938868ce7ffea1b1ffe9cf314cf2342f5466d88b。本次 scene_general_v3 的整理在 feat/unify-scene-v3-versioned-runner，服务器部署原提交 bf57b05；本轮 CI 修复之后需在服务器可连接时更新到新的云端提交。没有合并 main。

## 自动化验收

每次 push 或 pull_request 自动执行现有 Engineering contracts；冻结约束和原 v3 回归已加入常规 pytest：

- 安装声明的工程测试依赖与 CPU Torch；
- 从固定 Git 提交读取原 core、Prompt、冻结 v3 与已登记版本清单，检查没有修改；
- 检查每个版本源文件和 provenance 的 SHA256；
- Python 编译、完整 pytest（包含冻结契约与原生 v3 unittest）；
- Git 文件审计，禁止生成视频、预测、模型、机器配置和密钥进入提交。

原 bf57b05 的云端 CI 因工程测试环境缺少 requests 失败；本轮补齐 requests / scipy 测试依赖。服务器之前的 286 项 pytest 和 10 项原 v3 测试已经通过，不能因此声称原云端 CI 通过。无 GPU 的 CI 验收不代替真实 GPU 推理。

运行 python tools/check_scene_baseline.py 可重复源码约束检查。configs/workflows/frozen_contracts.json 从固定 Git 提交的文件建立独立 SHA256 快照，避免只改原源码和运行版本清单就通过校验。浅检出也能运行。审核时需要检查约束脚本、快照与 CI 本身未被绕过；它们同样不是权限控制。

## 开发规则

main 作为经过验收后合并的主线。实验在独立分支进行，明确版本 ID、原有输入与新运行目录。后续通过 PR 合并；合并前检查 CI 通过、core/Prompt 不变及 GPU 验收范围。没有新 GPU 验收的版本保持 pending 标记，兼容版默认仍使用原 I/O。

三个已经登记的版本清单不覆盖、不删除。新增算法版本放到独立源码目录并新增清单。运行时校验源指纹；续跑校验提交、配置、来源、权重和阶段产物，变化时创建新运行或回到原提交。开发分支名称与运行时 version ID 是两个概念。

## 当前硬限制

GitHub API 显示 main 与实验分支 protected=false。私有仓库当前套餐对 branch protection / rulesets 返回 HTTP 403，要求 GitHub Pro 或公开仓库。未改变仓库可见性或付费套餐。

因此 CI 可以发现违规和给出失败结果，但目前不能在 GitHub 侧强制阻止有写权限的人直接推送 main 或忽略失败合并。运行时的版本/来源校验已经生效。要强制 PR 与必需状态检查，需要启用支持私有仓库保护的套餐后配置主分支规则；管理员权限仍应谨慎使用。
