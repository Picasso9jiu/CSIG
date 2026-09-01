# 官方真实性审查提交清单

本清单对应组委会发布的《代码复现材料提交要求》。提交时不发送数据集；数据集由审查人员
从赛事官方渠道取得。

## 必交材料

1. **算法代码**：仓库中的 `model/`、`dataset/`、`utils/`、`lib/hais_ops/`，以及
   `train_temporal_memory.py`、`submit_challenge2.py`、`test2.py` 和 `scripts/`。
2. **模型文件**：`checkpoints/` 中最终提交使用的 M10、M26、M111 平均模型；
   `artifacts/m124_background_verifier_threshold065.pkl.gz` 是最终推理所需的冻结 verifier。
3. **算法报告**：`docs/ALGORITHM_REPORT_DREAM.md`。
4. **详细运行说明**：`README.md` 和 `docs/RUNNING_GUIDE.md`，包括环境、依赖版本、
   数据路径、快速核验、完整推理和从头训练命令。

训练/推理配置位于 `configs/`，固定训练链位于 `scripts/retrain_m26_chain.sh`，完整
checkpoint 推理位于 `scripts/run_m244_from_checkpoints.sh`。官方最终提交包及其审计材料
位于 `artifacts/`。

## 审查前本地检查

在已经取得官方测试 NPZ 的环境中，建议依次执行：

```bash
conda activate EV39
cd /path/to/EVSOD
git lfs pull

# 快速重建并核验最终 M244 提交包（不需要 GPU）
export DATA_ROOT="/path/to/测试集"
bash scripts/run_m244_fast.sh

# 核对模型与 artifact 哈希
sha256sum checkpoints/*.pt artifacts/m124_background_verifier_threshold065.pkl.gz
sha256sum artifacts/m233_base_submission.zip artifacts/m244_reference_submission.zip
```

快速路径应检查 31 个 TXT、事件顺序、五列格式、二值标签、总正样本数 87,706，且最终
归档 SHA-256 为：

```text
390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20
```

如果需要核对“可一键训练和测试”的入口，可在 GPU 环境设置 `TRAIN_DATA_ROOT`、
`TEST_DATA_ROOT`、`RUN_ROOT` 和 `OUTPUT_ROOT` 后运行：

```bash
bash scripts/run_train_and_test.sh
```

该命令会先检查 CUDA、HAIS_OP 和 spconv，再按 M4→M13→M15→M20→M26 顺序训练并运行
完整推理。它可能耗时较长，审查最终提交时优先使用快速路径。

## 邮件与压缩包命名

组委会邮箱：`mzj@nudt.edu.cn`。

邮件主题、附件压缩包和外层目录统一使用：

```text
评测平台测试集排名序号-队伍中文名称-评测平台英文名称-赛道几
```

例如：

```text
3-金睛杯-jinsight-赛道1.zip
```

排名序号、队伍中文名称、评测平台英文名称和赛道号必须以官方最终信息为准，不能沿用
示例值。仓库内的打包工具会校验四个字段并拒绝覆盖已有压缩包：

```bash
python scripts/package_review_materials.py \
  --rank <最终排名序号> \
  --team-cn "<队伍中文名称>" \
  --platform-en "<评测平台英文名称>" \
  --track <赛道号> \
  --output-dir review_packages
```

生成的 ZIP 包含源码、配置、环境文件、报告、运行说明、checkpoint 和必要审计 artifact，
自动排除数据集、Git 元数据、编译产物和运行输出。解压后可按 `EVSOD/README.md` 开始
复现。

仓库中的 `.pt`、`.pkl.gz` 和提交归档通过 Git LFS 管理。如果邮件系统限制附件大小，
不要删除模型文件；可将代码、报告和运行说明作为附件，并在邮件正文同时提供
<https://github.com/Picasso9jiu/CSIG> 及对应 Git LFS 下载地址，明确说明模型文件仍是
同一版本的审查材料。

## 提交前人工确认

- [ ] 排名、队伍名、平台英文名、赛道号已由官方信息确认；
- [ ] 邮件和 ZIP 文件名完全一致；
- [ ] ZIP 中没有 `train/`、`val/`、`test/` 的 NPZ 数据或其他数据集文件；
- [ ] Git LFS 文件已执行 `git lfs pull`，checkpoint 不是几十字节的指针文件；
- [ ] 快速复现和 `verify_submission.py` 均通过；
- [ ] 邮件包含四类材料，且在 2026-09-04 至 2026-09-08 期间发送至组委会邮箱。
