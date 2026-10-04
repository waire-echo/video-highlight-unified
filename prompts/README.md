# Prompt版本

运行文字集中在 `ora/v5`、`qwen/direct-v1`、`qwen/window-v1`、`qwen/global-event-v1` 和 `reze/yesno-v1`。动态构造器从旧代码逐函数提取，输出契约保持一致。

`ora/proposal-v1` 是旧请求样例与schema，含具体时长，只作历史参考，完整生成/整理代码未恢复。`ora/v5/reference` 是带来源说明的历史实际请求样例，不用于运行。官方未改源码中的Prompt由原Git commit版本化，位于 `pipelines/baselines/official`。

新Prompt创建新版本目录，配置指定bundle映射。实验保存每个文件hash；不要覆写已登记的旧版本，不使用final/final2命名。
