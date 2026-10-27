# 运行流程

[English](workflow.md) | [简体中文](workflow_zh.md)

所有阶段均使用通过 `requirements.txt` 安装的环境。数据整理和聚类在 CPU 上运行，向量构建和预测使用指定的 GPU。本地基础模型与 SAE 检查点路径在 [configs/paths.json](../configs/paths.json) 中配置。

## 四个脚本块

| 脚本 | 阶段 | 主要输出 |
| --- | --- | --- |
| `scripts/prepare_data.sh` | `prepare` → `cluster` | `train.pkl`、`test.pkl`、`clusters.json`、`anchors.pkl`、`test_clusters.pkl` |
| `scripts/build_global.sh` | `global` | `global.pt` |
| `scripts/build_local.sh` | `rank` → `pairs` → `local` | `ranked.json`、`accepted.json`、`rejected.json`、`local/` |
| `scripts/predict.sh` | `generate` → `evaluate` | `generation.json`、`metrics.json` |
| `scripts/run_all.sh` | 上述全部阶段 | 任务的完整输出 |

在项目根目录依次运行各块：

```bash
bash scripts/prepare_data.sh --tasks lamp4
bash scripts/build_global.sh --tasks lamp4 --gpus 0
bash scripts/build_local.sh --tasks lamp4 --gpus 0
bash scripts/predict.sh --tasks lamp4 --gpus 0
```

也可以用一个命令执行相同的完整流程：

```bash
bash scripts/run_all.sh --tasks lamp4 --gpus 0
```

使用 `--tasks lamp4 lamp5` 可选择多个任务，省略 `--tasks` 则依次运行四个任务。当前任务完成选定的全部阶段后，才会开始下一个任务。

| 任务名 | 配置文件 | 数据／输出子目录 |
| --- | --- | --- |
| `lamp4` | `configs/lamp4.json` | `LaMP_4` |
| `lamp5` | `configs/lamp5.json` | `LaMP_5` |
| `longlamp3` | `configs/longlamp3.json` | `product_review_temporal` |
| `longlamp4` | `configs/longlamp4.json` | `topic_writing_temporal` |

脚本使用本地输入和检查点，不会执行下载。选择起点前，请先从 [FestinaLente123/GLASS-artifacts](https://huggingface.co/datasets/FestinaLente123/GLASS-artifacts) 下载所需输入。

## 输入与输出

对于每个阶段输入，脚本先查找 `runs/<dataset>/`，再查找 `artifacts/<dataset>/`。新生成的输出始终写入 `runs/<dataset>/`，下载的参考文件保留在 `artifacts/` 中。

整理后的测试文件可命名为 `test.pkl` 或 `seen_test.pkl`。查找顺序为 `runs/<dataset>/test.pkl`、`runs/<dataset>/seen_test.pkl`、`artifacts/<dataset>/test.pkl`、`artifacts/<dataset>/seen_test.pkl`。其他阶段输入使用配置中的文件名。

用户和 anchor 抽样使用配置中的随机种子，默认为 42。从原始数据运行时不需要选择元数据。若要复用随产物提供的 LaMP 用户和 anchor 选择，请在 `prepare_data.sh` 或 `run_all.sh` 中添加 `--reference-selection`。启用后，数据整理阶段读取 `artifacts/<dataset>/<dataset>_users.json`，聚类阶段读取 `artifacts/<dataset>/<dataset>_target_indices.json`。只需提供本次选定阶段对应的选择文件。

运行开始前，脚本会检查所有选定阶段的输出是否为新路径。若输出文件或分片目录已经存在，会在计算开始前停止；脚本不会自动跳过已完成阶段。继续运行时，可用 `--from` 选择后续阶段作为起点。

## 从中间产物开始

每个脚本块均接受 `--from STAGE` 和 `--until STAGE`，起止阶段均包含在执行范围内。指定阶段必须属于该脚本块；`run_all.sh` 则接受完整流程中的任一阶段。例如，给 `build_local.sh` 添加 `--from pairs` 会依次执行 `pairs` 和 `local`，而 `--from pairs --until pairs` 只执行 Stage2。

下表列出了从各起点运行至所在块结束时所需的输入。中间输入下载到 `artifacts/<dataset>/`；评分使用本地预测阶段生成的 `runs/<dataset>/generation.json`。已存在于 `runs/<dataset>/` 中的其他输入无需再次下载。

| 起始阶段 | 脚本 | 所需输入（未注明路径时位于 `artifacts/<dataset>/`） |
| --- | --- | --- |
| `cluster` | `prepare_data.sh --from cluster` | `train.pkl`、`test.pkl` |
| `global` | `build_global.sh` | `train.pkl` |
| `rank` | `build_local.sh --from rank` | `anchors.pkl`、`global.pt` |
| `pairs` | `build_local.sh --from pairs` | `anchors.pkl`、`ranked.json`、`global.pt` |
| `local` | `build_local.sh --from local` | `accepted.json`、`global.pt` |
| `generate` | `predict.sh --from generate` | `test_clusters.pkl`、`global.pt`、`local/` |
| `evaluate` | `predict.sh --from evaluate` | 本地生成的 `runs/<dataset>/generation.json` |

全局向量块需要 `train.pkl` 和基础模型／SAE 检查点，不需要 anchors。局部向量块从 `anchors.pkl` 和已保存的全局方向开始。其他模型阶段使用基础模型和已保存的方向，不加载 SAE 检查点。若通过 `run_all.sh --from ...` 跨块继续运行，还需提供后续块所需的输入，例如预测阶段的 `test_clusters.pkl`。

### 从整理好的用户数据划分开始

下载整理好的数据划分，再运行聚类，并按照随机种子抽取 anchors：

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/train.pkl' \
            'artifacts/LaMP_4/test.pkl' \
  --local-dir .
bash scripts/prepare_data.sh --tasks lamp4 --from cluster
```

随后继续运行 `build_global.sh`、`build_local.sh` 和 `predict.sh`。这条路径跳过原始数据整理，重新构建其后的所有中间结果。

若要复用随产物提供的 anchor 选择，请另外下载 `artifacts/LaMP_4/LaMP_4_target_indices.json`，并在聚类命令中添加 `--reference-selection`。

### 从 Stage1 开始

下载 anchors 和全局向量，再重新构建排序、样本对和局部方向：

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/anchors.pkl' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from rank --gpus 0
```

### 从 Stage2 开始

复用现成的排序结果，重新生成对比样本对和局部方向：

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/anchors.pkl' \
            'artifacts/LaMP_4/ranked.json' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from pairs --gpus 0
```

### 从局部方向构建开始

下载 Stage2 保留的样本对和全局向量：

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/accepted.json' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from local --gpus 0
```

### 从预测开始

使用现成向量和测试分簇运行 Stage3 与评分：

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/test_clusters.pkl' \
            'artifacts/LaMP_4/global.pt' \
            'artifacts/LaMP_4/local/*' \
  --local-dir .
bash scripts/predict.sh --tasks lamp4 --gpus 0
```

若本地已生成 `runs/LaMP_4/generation.json`，可单独重新评分，无需再次进行模型推理：

```bash
bash scripts/predict.sh --tasks lamp4 --from evaluate
```

其他任务按上表替换任务名和产物子目录即可。

## 进度、日志与执行计划

阶段标签会标明当前任务和操作。交互式终端显示进度条；重定向后的日志包含阶段开始／结束消息和定期进度更新。只有进程成功退出，包括完成输出写入和分片合并后，才表示该阶段完成。

每次启动都会将阶段日志和展开后的配置保存在 `runs/pipeline_<timestamp>/`。模型工作进程的日志也会保存到阶段输出旁的 `*.shards/` 目录。某一阶段失败时，流程停止并给出日志位置。Ctrl-C 会停止当前阶段及其工作进程。

使用 `--dry-run` 可显示选定阶段、解析后的输入和输出，不加载模型，也不启动计算：

```bash
bash scripts/build_local.sh --tasks lamp4 --from pairs --gpus 0 1 --dry-run
```

## 并行执行与参数

传入多个 GPU 编号，即可按完整用户分片运行：

```bash
bash scripts/build_local.sh --tasks lamp4 --gpus 0 1 2 3 4 5 6 7
```

每个工作进程加载一份模型并处理完整用户；所有工作进程成功后，主进程按输入顺序合并输出。Stage2 工作进程仍可访问其他用户的完整负例池。按用户分片不会自动增大配置中的 batch 大小。

任务参数保存在 `configs/<task>.json` 中。每个阶段读取自身配置节以及 `common`；修改任务配置即可调整 batch 大小或 steering 强度等参数。运行单个阶段时，Python 命令行还支持 `--set KEY=VALUE`，其中值可为 JSON 数值、列表、布尔值和 null：

```bash
python -m glass rank --config configs/lamp4.json --gpus 0 1 \
  --set input_path=artifacts/LaMP_4/anchors.pkl \
  --set global_vectors_path=artifacts/LaMP_4/global.pt \
  --set output_path=runs/LaMP_4/ranked.json
```

## 评分

ROUGE-1 和 ROUGE-L 使用 `rouge-score`，并启用词干化。汇总结果是逐样本 F1、precision 和 recall 的算术平均值，范围为 0–1。空生成文本的 ROUGE 得分为零。

METEOR 是可选项（在单阶段评分命令中设置 `--set meteor=true`），使用 NLTK 分词和 WordNet。离线运行前请安装所需资源：

```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
```

生成和评分阶段不会自动下载这些资源。
