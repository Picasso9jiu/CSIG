# DREAM 运行说明

本文是 EVSOD/DREAM 发布版的详细运行手册。提交审查不需要发送数据集；请审查人员从
赛事官方渠道取得数据，并按本文设置 `DATA_ROOT`。所有命令均在 WSL/Ubuntu 的 Bash
中运行，Windows PowerShell 仅用于启动 WSL，不直接执行 `.sh` 脚本。

## 一、材料与目录

从 GitHub 获取代码：

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
git clone https://github.com/Picasso9jiu/CSIG.git EVSOD
cd EVSOD
git lfs pull
```

拉取完成后，发布目录中应至少存在：

```text
EVSOD/
├── README.md
├── requirements.txt
├── requirements-cuda111.txt
├── environment.yml
├── configs/
├── dataset/
├── model/
├── utils/
├── lib/hais_ops/                 # 自定义算子源码
├── checkpoints/                  # 发布模型
├── artifacts/                    # verifier、审计数据和官方提交包
├── scripts/
├── docs/ALGORITHM_REPORT_DREAM.md
└── docs/RUNNING_GUIDE.md
```

数据集不随仓库提供。测试数据目录应为：

```text
/path/to/测试集/
└── test/
    ├── test_000.npz
    ├── ...
    └── test_030.npz
```

训练/验证数据目录应为：

```text
/path/to/训练集、验证集/
├── train/
├── val/
└── val_Challenge2.py        # 若官方数据包提供
```

## 二、环境配置

### 2.1 快速提交包核验

快速路径不重新运行神经网络，只使用 Python、NumPy 和仓库中的冻结 artifact 重建
M244。因此它不需要 GPU、CUDA、spconv 或 HAIS_OP；使用完整 EV39 环境仍最便于统一审查。

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
python -m pip install --upgrade pip
python -m pip install numpy==1.23.5
```

### 2.2 完整 checkpoint 推理和重训

完整路径需要 NVIDIA GPU、CUDA 11.1 兼容运行时和 Python 3.9。推荐使用发布的依赖文件：

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda111.txt
```

`requirements-cuda111.txt` 固定了 PyTorch 1.9.1+cu111、torchvision 0.10.1+cu111，
并引用 `requirements.txt` 中的 NumPy、PyYAML、tqdm、spconv 等版本。也可以使用：

```bash
conda env create -f environment.yml
conda activate EV39
```

在仓库根目录编译 HAIS_OP：

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build
cd lib/hais_ops
python setup.py build_ext develop
cd ../..
```

每次新开终端后设置运行时路径并检查依赖：

```bash
conda activate EV39
export PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
python -c "import torch, HAIS_OP; import spconv.pytorch; print('cuda:', torch.cuda.is_available()); print('HAIS_OP: ok')"
```

只有在 CUDA 输出为 True 且 HAIS_OP 导入成功后，才运行完整 checkpoint 推理或重训。

## 三、路径 A：快速重建官方 M244

这是官方真实性审查最推荐的路径。它不训练、不读标签、不依赖平台接口，直接根据已发布
M233 基础包、M243 冻结审计报告和公开测试事件重建 M244。

```bash
cd /path/to/EVSOD
conda activate EV39
export DATA_ROOT="/path/to/测试集"
bash scripts/run_m244_fast.sh
```

脚本默认读取 `$DATA_ROOT/test/`，输出：

```text
outputs/m244_empty_top4_raw_test.zip
```

预期核验结果：

```text
valid_txt_files: 31
total_events: 2265422
total_positive_events: 87706
sha256: 390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20
```

也可直接核验仓库内的官方归档：

```bash
python -B scripts/verify_submission.py \
  --zip artifacts/m244_reference_submission.zip \
  --test-root "$DATA_ROOT/test" \
  --m244
```

该命令会检查 31 个 TXT 的文件列表、行数、x/y/t/p 顺序、二值 label、总正样本数和
M244 SHA-256。隐藏标签不在仓库中，所以这里核验的是提交包真实性和格式，而不是平台
Score 的重新计算。

## 四、路径 B：从发布 checkpoint 运行完整测试

该路径不使用 M233 参考提交包，而是从 M10、M26、M111 平均权重和 M124 verifier 重新
生成预测，再合并为 M244。它需要完整 EV39、CUDA 和已编译 HAIS_OP，运行时间明显长于
路径 A。

```bash
cd /path/to/EVSOD
conda activate EV39
export PROJECT_DIR="$(pwd)"
export DATA_ROOT="/path/to/测试集"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
bash scripts/run_m244_from_checkpoints.sh
```

脚本会：

1. 用 `split_m244_domains.py` 按事件数和极性比例划分四个无标签域；
2. 为每个域生成相应 M124 阈值 artifact；
3. 调用 `submit_challenge2.py` 运行 M10/M26/M111 推理和固定后处理；
4. 合并各域 TXT，审计 M243 原始分数；
5. 调用 `rebuild_m244.py` 生成并核验最终 M244 ZIP。

输出目录默认为 `outputs/m244_from_checkpoints/`。如需使用其他发布模型，可通过环境变量
覆盖，而不需要改源码：

```bash
export M10_CHECKPOINT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
export M26_CHECKPOINT="$PROJECT_DIR/checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt"
export M111_CHECKPOINT="$PROJECT_DIR/checkpoints/m111_phase_specialist_seed72_73_76_average.pt"
export M124_SOURCE_ARTIFACT="$PROJECT_DIR/artifacts/m124_background_verifier_threshold065.pkl.gz"
```

脚本会拒绝覆盖已有输出目录，以免误删先前的复现结果。

## 五、路径 C：一键从头训练并测试

`scripts/run_train_and_test.sh` 是完整训练/测试入口。它按发布训练链依次执行 M13、
M15、M20、M26，然后自动找到 M26 epoch 003 checkpoint，并调用完整 M244 测试流程。
这是审查所需的“一键训练和测试”入口，训练时间取决于 GPU 和磁盘速度，可能需要较长
时间；只核验最终提交时应优先使用路径 A。

```bash
cd /path/to/EVSOD
conda activate EV39
export TRAIN_DATA_ROOT="/path/to/训练集、验证集"
export TEST_DATA_ROOT="/path/to/测试集"
export RUN_ROOT="$PWD/outputs/retrain_m26_one_click"
export OUTPUT_ROOT="$PWD/outputs/final_test_one_click"
bash scripts/run_train_and_test.sh
```

输入目录必须同时包含：

```text
$TRAIN_DATA_ROOT/train/
$TRAIN_DATA_ROOT/val/
$TEST_DATA_ROOT/test/
```

脚本默认不覆盖已有 `RUN_ROOT` 或 `OUTPUT_ROOT`。训练完成后会打印：

```text
M26 checkpoint: .../epoch_003_seed53.pt
M244 package: .../m244_empty_top4_raw_test.zip
```

请先用 `scripts/verify_submission.py` 检查生成 ZIP，再与官方归档或平台提交结果对照。
由于不同 CUDA、spconv 和 GPU 算子可能产生极小浮点差异，完整网络重算不保证 ZIP 字节
级等于官方归档；路径 A 才是官方包的字节级复现路径。

## 六、辅助核验

### 6.1 M111 平均模型

```bash
python -B scripts/build_m111_average.py \
  --checkpoint checkpoints/m111_phase_seed72_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed73_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed76_epoch_001.pt \
  --output /tmp/m111_average.pt \
  --reference checkpoints/m111_phase_specialist_seed72_73_76_average.pt
```

预期输出 `reference_model_state: ok`。平均过程对模型张量使用 float64，之后恢复原始
dtype；序列化元数据差异不会影响张量核验。

### 6.2 模型和 artifact 哈希

```bash
sha256sum checkpoints/*.pt artifacts/m124_background_verifier_threshold065.pkl.gz
sha256sum artifacts/m233_base_submission.zip artifacts/m244_reference_submission.zip
```

逐文件期望值见 `docs/CHECKSUMS.sha256`，模型角色和最终使用关系见
`checkpoints/MODEL_MANIFEST.json`。

## 七、常见问题

**1. `git lfs` 不存在或 artifact 很小。**

在 WSL 安装 `git-lfs`，执行 `git lfs install` 和 `git lfs pull`。LFS 指针文件通常只有
几百字节，不能用于运行。

**2. 找不到测试文件。**

`DATA_ROOT` 必须指向包含 `test/` 子目录的“测试集”目录，而不是直接指向 `test/`。
训练入口同理要求 `TRAIN_DATA_ROOT` 下同时有 `train/` 和 `val/`。

**3. 在 PowerShell 中运行 `.sh` 失败。**

请先进入 WSL/Ubuntu，再在 Bash 中执行本文命令。路径应使用 `/mnt/d/...` 形式。

**4. `CUDA is required` 或 `HAIS_OP` 导入失败。**

确认已激活 EV39、CUDA 驱动可见、HAIS_OP 已在 `lib/hais_ops` 编译，并重新设置
`PYTHONPATH` 和 `LD_LIBRARY_PATH`。快速路径不需要这些组件。

**5. 平台 Score 无法本地计算。**

这是正常现象：官方隐藏测试标签不公开。仓库只能复现提交包和检查事件级格式；平台分数
需由组委会评测服务计算。

## 八、审查材料对应表

| 官方要求 | 本仓库材料 |
| --- | --- |
| 算法代码 | `model/`、`dataset/`、`utils/`、`lib/hais_ops/`、`train_temporal_memory.py`、`submit_challenge2.py`、`test2.py` |
| 模型文件 | `checkpoints/` 中 M10、M26、M111 平均模型及其辅助 checkpoint |
| 算法报告 | `docs/ALGORITHM_REPORT_DREAM.md` |
| 运行说明 | `README.md`、本文档 `docs/RUNNING_GUIDE.md` |
| 配置和训练脚本 | `configs/`、`scripts/retrain_m26_chain.sh`、`scripts/run_train_and_test.sh` |
| 最终提交审计材料 | `artifacts/`、`scripts/verify_submission.py`、`scripts/rebuild_m244.py` |

数据集不包含在仓库或提交材料中，符合组委会“无需发送数据集”的要求。
