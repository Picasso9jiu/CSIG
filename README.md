# EVSOD：DREAM 最终方案复现

本仓库是 EV-UAV Challenge 2 最终提交方案的独立发布版本。方案对外名称为
**DREAM**（**D**ensity-**R**outed **E**vent **A**ttention **M**emory，
密度路由事件注意力记忆模型）。

内部实验编号和官方归档编号为 **M244**。该编号仅用于将仓库中的 checkpoint、
审计材料和官方提交包一一对应；方法名称、文档和对外介绍统一使用 DREAM。

DREAM 面向事件相机小目标检测：它先从整段事件流中提取无需标签即可获得的密度和
极性统计量，选择合适的时序专家，再使用双向时序记忆、时间注意力、相位互补融合和
保守的组件/轨迹后处理生成逐事件标签。最终的 M244 归档在 M233 主预测链基础上，
通过冻结的 M243 原始分数审计执行一次极小范围的无标签恢复。

官方平台保留的 M244 隐藏测试结果为：

| Score | Pd | Fa | IoU | Acc |
| ---: | ---: | ---: | ---: | ---: |
| **0.9347** | 0.9318 | 5.56e-06 | 0.9065 | 0.9694 |

隐藏测试集标签不公开，也不包含在本仓库中，因此上述平台分数无法本地重算；但是提交
TXT 的事件顺序、逐行字段、固定无标签决策链以及官方 ZIP 包都可以完整复现和审计。

## 官方真实性审查材料对照

本仓库按组委会要求同时提供四类材料，且不包含数据集：

| 官方要求 | 本仓库对应内容 |
| --- | --- |
| ① 算法代码 | `model/`、`dataset/`、`utils/`、`lib/hais_ops/`、`train_temporal_memory.py`、`submit_challenge2.py`、`test2.py` |
| ② 模型文件 | `checkpoints/` 中的 M10、M26、M111 平均模型，以及 `artifacts/m124_background_verifier_threshold065.pkl.gz` |
| ③ 算法报告 | [`docs/ALGORITHM_REPORT_DREAM.md`](docs/ALGORITHM_REPORT_DREAM.md) |
| ④ 运行说明 | [`docs/RUNNING_GUIDE.md`](docs/RUNNING_GUIDE.md)、[`docs/REVIEW_CHECKLIST.md`](docs/REVIEW_CHECKLIST.md) 和本文 README |

训练/推理配置、固定参数和阶段间 checkpoint 对应关系分别保存在 `configs/`、
`scripts/retrain_m26_chain.sh`、`scripts/run_m244_from_checkpoints.sh` 和
`checkpoints/MODEL_MANIFEST.json` 中。`scripts/run_train_and_test.sh` 提供从训练到
测试的一键入口；`scripts/package_review_materials.py` 可按官方命名规则生成审查压缩包。

提交材料中不发送数据集。审查人员需自行从赛事官方渠道取得原始 NPZ，并按运行说明设置
训练集、验证集和测试集路径。

官方提交前的材料、命名和逐项核验清单见
[`docs/REVIEW_CHECKLIST.md`](docs/REVIEW_CHECKLIST.md)。需要生成正式审查压缩包时，使用
该清单末尾的 `scripts/package_review_materials.py` 命令，并将排名、队伍名称、平台英文名
和赛道号替换为官方最终信息。

## 获取代码与大文件

请先在 WSL/Ubuntu 中安装 Git 与 Git LFS、克隆仓库并拉取 LFS 文件。后续所有命令都在
克隆得到的 EVSOD 仓库根目录执行：

~~~bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install

git clone https://github.com/Picasso9jiu/CSIG.git EVSOD
cd EVSOD
git lfs pull
~~~

## 环境配置

本文中的 shell 命令均在 **WSL/Ubuntu 的 Bash** 中验证，不是在 Windows PowerShell
中直接运行。发布版本使用 Python 3.9、PyTorch 1.9.1 + CUDA 11.1、torchvision
0.10.1、spconv-cu111、NumPy 1.23.5 和 CUDA 11.x Toolkit。环境名称固定为 EV39。

若本机尚未创建 EV39，请在仓库根目录执行（快速核验只需要其中的 Python 3.9 和 NumPy；
完整推理/重训再安装 CUDA 依赖）：

~~~bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39

python -m pip install --upgrade pip
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt
~~~

也可以使用仓库提供的 [`environment.yml`](environment.yml) 一次创建完整环境：

~~~bash
conda env create -f environment.yml
conda activate EV39
~~~

上面的 EV39 环境已经足以执行下方“快速复现官方提交包”。快速脚本只重建和核验已发布
的 ZIP，不重新运行网络，因此不使用 GPU，也不需要编译 HAIS_OP。

若要执行后文的完整 checkpoint 推理或从头重训，再在仓库根目录额外安装编译依赖并
编译 HAIS_OP：

~~~bash
conda activate EV39
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build
cd lib/hais_ops
python setup.py build_ext develop
cd ../..

export PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
python -c "import torch, HAIS_OP; import spconv.pytorch; print(torch.cuda.is_available(), 'HAIS_OP: ok')"
~~~

完整 checkpoint 推理和从头重训必须满足上面命令输出 True 和 HAIS_OP: ok。每次新开
终端后再次执行完整推理时，需要重新激活 EV39 并设置同一组 PROJECT_DIR、PYTHONPATH
和 LD_LIBRARY_PATH。

## 快速复现官方提交包

这是推荐的查验路径。它不需要 GPU，也不需要重新训练或重新跑网络：脚本以版本化的
M233 基础提交包和 M243 冻结审计报告，精确重建官方 M244 ZIP，再对每一行输出与
公开测试事件逐项核验。

~~~bash
conda activate EV39
export DATA_ROOT="/path/to/测试集"
bash scripts/run_m244_fast.sh
~~~

成功时应看到：

~~~text
valid_txt_files: 31
total_events: 2265422
total_positive_events: 87706
sha256: 390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20
~~~

生成的提交包位于：

~~~text
outputs/m244_empty_top4_raw_test.zip
~~~

也可以单独验证仓库中保存的官方归档包：

~~~bash
python scripts/verify_submission.py \
  --zip artifacts/m244_reference_submission.zip \
  --test-root "$DATA_ROOT/test" \
  --m244
~~~

快速复现使用以下已版本化的材料：

~~~text
artifacts/m233_base_submission.zip        # 冻结的 M233 主预测提交包
artifacts/m243_raw_scores/*.npy           # 仅含原始模型分数，不含测试标签
artifacts/m243_test_actual.json           # 固定的无标签 Top-4 审计结果
artifacts/m244_reference_submission.zip   # 官方 M244 归档包
~~~

其中 scripts/audit_m243_raw_scores.py 可从原始分数和公开输入事件重新计算审计报告；
scripts/rebuild_m244.py 会在写出 ZIP 前核验每个 TXT 的行数、x/y/t/p 字段顺序和
二值标签范围。

## 官方审查压缩包

组委会要求的邮件/压缩包命名格式、四类材料和发送前检查项见
[`docs/REVIEW_CHECKLIST.md`](docs/REVIEW_CHECKLIST.md)。确认官方排名、队伍中文名称、
平台英文名称和赛道号后，可从仓库根目录生成不含数据集的材料包：

~~~bash
python scripts/package_review_materials.py \
  --rank <最终排名序号> \
  --team-cn "<队伍中文名称>" \
  --platform-en "<评测平台英文名称>" \
  --track <赛道号> \
  --output-dir review_packages
~~~

工具会拒绝覆盖同名文件，并在压缩包内写入
`EVSOD/review_materials_manifest.json`（逐文件大小和 SHA-256）。它会自动排除数据集
NPZ、Git 元数据、编译产物和运行输出；生成后仍应按清单人工确认文件名和收件邮箱。
模型与 verifier 通过 Git LFS 管理；若邮件附件大小受限，请保留全部模型材料，并在邮件
正文提供本仓库地址及对应 LFS 下载说明，不要用缺失权重的精简包代替完整审查材料。

## DREAM 方法详解

### 1. 整体流程

输入为按时间排序的原始事件序列，每个事件保留五列：

~~~text
x y t p label
~~~

其中 x、y、t、p 必须与原始 NPZ 中的事件逐行完全一致，DREAM 只预测最后一列
label。完整推理链可以概括为：

~~~text
原始事件流
  -> 整段无标签统计（事件量、极性比例等）
  -> 密度路由：M10 低事件量专家 或 M26 主时序专家
  -> M26 原相位分支 + M111 半时间 bin 相位分支
  -> 0.75/0.25 分数融合
  -> P6 分数阈值校准
  -> P0/P0c 组件过滤与高置信恢复
  -> P18 弱轨迹恢复 + P32 轨迹质量加分
  -> M124 持续背景组件核验
  -> M244 极高密度域的冻结无标签 Top-4 恢复
  -> 逐事件 TXT / ZIP
~~~

这套设计的原则是：先让时序模型尽可能保留真实小目标的连续响应，再通过结构化规则
去除孤立噪声和持续背景；路由和阈值选择只依赖整段视频中可观测的事件统计，不读取
测试标签。

### 2. 密度路由与双专家预测

不同视频的事件量相差很大。低事件量视频的稀疏性更强，直接使用高密度时序模型容易
使有效候选被过度平滑；高事件量视频则更需要跨时间的上下文约束。因此 DREAM 将
推理分为两个专家：

| 输入域 | 选择的专家 | 设计目的 |
| --- | --- | --- |
| 整段事件数不超过 30,000 | M10 低事件量专家 | 保留稀疏小目标响应，避免时序聚合过强 |
| 整段事件数超过 30,000 | M26 时序主专家 | 利用连续运动和上下文，抑制高密度背景噪声 |

该路由由输入事件数固定决定，在未知测试视频上无需标签即可直接运行。两个专家的
checkpoint 均已随仓库发布，完整 checkpoint 推理脚本会自动根据这一规则分组运行。

### 3. M26 双向时序记忆与时间注意力

M26 是 DREAM 的主干高密度专家。它将连续事件划分为时间 bin，在稀疏体素特征上维护
跨 bin 的记忆状态，并同时利用前向和反向时间上下文。这样，当前 bin 的小目标候选
不仅依赖局部空间响应，也能得到前后相邻时间段的支持。

主干中包含三个关键约束：

1. **双向时序记忆**：前向和反向特征融合，降低单向传播造成的起始/结束边界漏检。
2. **时间自注意力**：对长序列中更有信息的历史和未来 bin 赋予更高权重，避免把所有
   时间片等权平均。
3. **有界目标流对齐头**：对相邻时间特征施加受限位移对齐，使连续运动目标更容易形成
   稳定响应，同时限制异常大位移带来的错误传播。

这些模块主要提升连续小目标的召回和时序一致性，为后续的组件和轨迹规则提供更可靠的
候选分数。

### 4. 相位互补分支与分数融合

固定时间 bin 的边界会导致同一目标在相邻 bin 中被切开。为减少这种相位敏感性，
DREAM 引入 M111 相位专家：

1. M111 由三个独立的一轮相位专家在参数空间进行 float64 平均得到；
2. 它使用相对主分支平移 25 的半时间-bin 输入，观察与原分箱错开的时间相位；
3. 最终分数按原相位 M26 占 0.75、相位专家占 0.25 的比例融合。

原相位分支保持主模型的稳定性；半相位分支只以较小权重补足处于时间边界附近的弱响应，
从而减少因为分箱位置不同造成的漏检。

### 5. 分数校准、组件约束与弱轨迹恢复

网络输出不是直接二值化。DREAM 使用一组固定的结构化后处理，目标是在 Pd、Fa 和 IoU
之间取得平衡。

**P6：基于输入统计的分数阈值校准**

- 低事件量域使用 0.718；
- 中等事件量域使用 0.728；
- 高事件量域再按整段极性少数类比例是否低于 0.20，分别使用 0.722 或 0.724。

这些阈值只由事件总量和极性比例决定，不依赖视频名称或标签。

**P0/P0c：时空连通组件过滤与高置信恢复**

- 在空间半径 2、时间 bin 大小 50 的邻域中建立候选组件；
- 剔除事件数少于 3 或持续时间不足 5 个 bin 的短小孤立组件；
- 对分数不低于 0.95 的高置信事件执行保守恢复，避免组件过滤误删极强真实响应。

这一步主要压制离散噪声和短促背景闪烁，同时保留少数强目标证据。

**P18：单事件弱轨迹恢复**

- 从分数至少为 0.53 的候选中寻找时间连续、空间可链接的弱轨迹；
- 使用空间半径 5、最大链接距离 8、最多跨 1 个时间 bin 的约束；
- 轨迹至少覆盖 4 个 bin 时，仅恢复其中最可信的事件。

P18 不会整段放宽阈值，而是只恢复具有运动连续性支撑的局部弱响应，重点服务于 Pd。

**P32：轨迹质量加分**

- 从分数至少为 0.60 的候选开始；
- 需要至少 4 个连续 bin、至少 3 个种子组件，并满足最大运动残差 2；
- 对符合条件的候选加 0.010 分，分数上限限制为 0.97。

该规则让结构可靠的运动轨迹更容易跨越最终阈值，而对静态、孤立或运动不一致的背景
候选保持保守。

### 6. M124 持续背景组件核验与密度域阈值

高密度视频中，持续出现的背景热区可能在多个 bin 内形成看似连续的组件。M124 是一个
冻结的背景组件核验器，用于对这类持续背景组件进行额外删除。为避免同一阈值在不同
事件域下过强或过弱，DREAM 只根据整段视频的可观测统计选择 M124 阈值：

| 域 | 路由规则（仅输入事件） | M124 阈值 |
| --- | --- | ---: |
| 低/中密度 | 事件数不超过 200,000 | 0.94 |
| H1 高密度域 | 事件数超过 200,000，且极性少数类比例低于 0.20 | 0.62 |
| 其他 H2 高密度域 | 剩余高密度视频 | 0.65 |
| 极高 H2 域 | 其他 H2 中事件数不低于 500,000 | 0.63 |

这里的阈值路由是固定规则。完整推理脚本会先通过
scripts/split_m244_domains.py 生成各域清单，再为每个域写入对应阈值的 M124 artifact，
最后按原始事件顺序合并所有 TXT。

### 7. M244 的最终无标签恢复

M244 是 M233 的保守扩展，而不是重新训练的模型。对极高 H2 域视频，它在每个视频中：

1. 找到四个原本没有正预测的 50-unit 时间 bin；
2. 按冻结原始分数挑选其中分数最高的一个事件；
3. 将该事件由 0 恢复为 1。

在官方公开测试输入上，该规则只恢复了 test_022 和 test_023 中各 4 个事件，共 8 个
事件；其余事件与 M233 完全相同。所有候选索引均由
scripts/audit_m243_raw_scores.py 从冻结原始分数和输入事件生成，并保存为审计报告。

为保证官方归档可逐字节追溯，快速复现脚本会按该冻结报告重放这 8 个恢复操作。该审计
材料不包含隐藏测试标签；它的作用是让最终提交包的有限修改可检查、可复算，而不是在
复现时重新试探参数。

更多发布来源、文件哈希和 M244/M233/M243 的关系见
[docs/M244_RELEASE_NOTES.md](docs/M244_RELEASE_NOTES.md)。完整实验历史见
[note.md](note.md)，其中保留了成功和失败的尝试，供后续研究避免重复无效方向。

## 数据集目录与处理方式

数据集不随本仓库重新分发，请从赛事或官方渠道取得原始 NPZ 文件，并保持原始目录结构。
事件解析、稀疏体素化、时间采样和数据加载代码均在 dataset/ 下；不需要预先生成图像帧。

~~~text
训练集、验证集/
|-- train/                         # 官方训练集 NPZ
|-- val/                           # 官方验证集 NPZ
|-- val_Challenge2.py

测试集/
|-- test/                          # test_000.npz ... test_030.npz
~~~

完整 checkpoint 推理脚本读取 DATA_ROOT/test/；重训链读取 DATA_ROOT/train/ 和
DATA_ROOT/val/。提交 TXT 的格式为：

~~~text
x y t p label
~~~

scripts/verify_submission.py 会核验每个文件的行数、字段数、x/y/t/p 原始顺序和 label
是否为二值，防止因格式或事件顺序错误导致平台评分失效。

## 使用 checkpoint 完整推理

该路径会从发布的 M10、M26、M111 checkpoint 重建 M233 分数流，按固定无标签路由执行
M124，然后重新审计 M243 原始分数并构造 M244。它比快速复现慢，但不复用参考提交包，
需要已配置的 CUDA 环境。

~~~bash
conda activate EV39
export DATA_ROOT="/path/to/测试集"
bash scripts/run_m244_from_checkpoints.sh
~~~

输出写入：

~~~text
outputs/m244_from_checkpoints/
~~~

该脚本会保存每个视频的事件数、极性比例和所选 M124 阈值的 manifest。由于 CUDA、
spconv 和浮点算子版本可能带来极小数值差异，完整推理输出不承诺与官方 ZIP 字节级
一致；官方包的字节级核验以“快速复现官方提交包”路径为准。

## Checkpoint 与 M111 平均权重核验

所有发布 checkpoint、artifact 和 SHA-256 值见
[docs/CHECKSUMS.sha256](docs/CHECKSUMS.sha256)。完整推理需要 M10、M26、M111 和
M124 verifier；M4/M13/M15/M20 用于说明和复算 M26 的训练链。

三个 M111 seed checkpoint 也随仓库提供，可重新验证平均过程：

~~~bash
python scripts/build_m111_average.py \
  --checkpoint checkpoints/m111_phase_seed72_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed73_epoch_001.pt \
  --checkpoint checkpoints/m111_phase_seed76_epoch_001.pt \
  --output /tmp/m111_average.pt \
  --reference checkpoints/m111_phase_specialist_seed72_73_76_average.pt
~~~

命令应输出：

~~~text
reference_model_state: ok
~~~

平均时先将模型张量转为 float64，完成逐参数平均后再转回原 dtype。新写 checkpoint 的
序列化元数据可能使字节哈希不同，但每个模型张量应与仓库中的平均 checkpoint 一致。

## 从头重训主模型链

M26 主模型的版本化训练链为：

~~~text
M4+DACC+M5 -> M13 epoch 003 -> M15 epoch 008 -> M20 epoch 003 -> M26 epoch 003
~~~

所有训练命令、固定超参数和 epoch 选择已写入
scripts/retrain_m26_chain.sh。它从仓库内的 M4 checkpoint 起步，训练数据只使用 train/
目录，不混入验证或隐藏测试标签；该流程耗时较长。

~~~bash
conda activate EV39
export PROJECT_DIR="$(pwd)"
export DATA_ROOT="/path/to/训练集、验证集"
export RUN_ROOT="$PROJECT_DIR/outputs/retrain_m26"
bash scripts/retrain_m26_chain.sh
~~~

训练完成后，需先独立评估生成的 epoch-003 M26 checkpoint，再将其用于完整推理。
M10 和三个一轮 M111 相位专家均作为冻结专家随发布版提供，因此最终 DREAM/M244
管线可以直接按 checkpoint 完整复算。

## 仓库结构

~~~text
checkpoints/       M4/M10/M13/M15/M20/M26/M111 发布权重
artifacts/         M124 verifier、M233 基础包、M243 审计输入、官方 M244 ZIP
configs/           固定 YAML 与命令行覆盖配置
dataset/           官方 NPZ 解析、稀疏化和时间采样
lib/hais_ops/      CUDA 扩展源码（需本地编译）
model/             EV-SpSegNet 与时序记忆网络
utils/             推理、后处理、M124 与评估工具
scripts/           快速重建、完整推理、验证和重训入口
note.md             全部实验日志与结果记录
~~~

## 复现边界与致谢

本仓库公开的内容足以复现和审计最终提交包；平台隐藏标签和对应平台评分不在仓库内。
请使用公开输入数据、发布 checkpoint 和固定脚本完成复现，不应将平台隐藏标签用于
调参或训练。

该发布版本基于 ICCV 2025 EV-UAV 官方 baseline 继续开发。数据集、官方 baseline 和
引用信息见 [NOTICE.md](NOTICE.md)。
