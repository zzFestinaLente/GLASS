# From Profiles to Steering Vectors: Global Sparse Priors and Local Semantic Calibration for Personalized Text Generation

[![arXiv](https://img.shields.io/badge/arXiv-2607.21620-b31b1b?logo=arxiv)](https://arxiv.org/abs/2607.21620)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Artifacts-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/FestinaLente123/GLASS-artifacts)

[English](README.md) | [简体中文](README_zh.md)

GLASS (Global–Local Activation Steering with Sparse priors) builds a global style direction from a user's historical responses and a bank of local contrastive directions from clusters of that user's history. At inference, it adds the global direction and the selected local direction to two residual-stream layers of a frozen language model.

<p align="center">
  <img src="assets/glass_overview.png" alt="GLASS framework overview" width="600">
</p>

## Preparation

### Environment

Use one Python 3.12 environment for all stages. Install the dependencies in `requirements.txt`, including NumPy 2.2.6, PyTorch 2.8.0+cu126 and Transformers 4.57.1.

```bash
conda create -n glass python=3.12 pip -y
conda activate glass
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

ROUGE-1 and ROUGE-L use `rouge-score`, which is included in the dependencies. To also calculate METEOR (`--set meteor=true` in the single-stage evaluation command), download the NLTK resources before running offline:

```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
```

Generation and evaluation do not download these resources automatically.

### Models

The backbone is **Meta-Llama-3-8B base**. Use the pretrained 32× SAE from `EleutherAI/sae-llama-3-8b-32x-v2`, selecting its `layers.19` checkpoint.

| Component | Download |
| --- | --- |
| Backbone | [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) · [Model files](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main) |
| SAE layer 19 | [Checkpoint files](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x-v2/tree/main/layers.19) |

Edit [configs/paths.json](configs/paths.json) to point to your local checkpoints:

```json
{
  "model_path": "models/Meta-Llama-3-8B",
  "sae_model_path": "models/sae-llama-3-8b-32x-v2/layers.19"
}
```

### Datasets

For a new experiment from raw data, download the official datasets below and use the directory layout shown. The Hugging Face artifact repository provides processed intermediate files and metadata; raw benchmark files are obtained separately from the official sources below. To use precomputed intermediate artifacts, proceed directly to Quick start.

| Task | `--tasks` | Official raw data | Destination |
| --- | --- | --- | --- |
| LaMP-4 | `lamp4` | [Training files](https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_4/train/) · [Development files](https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_4/dev/) | `artifacts/LaMP_4/raw/` |
| LaMP-5 | `lamp5` | [Training files](https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_5/train/) · [Development files](https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_5/dev/) | `artifacts/LaMP_5/raw/` |
| LongLaMP-3 | `longlamp3` | [Product review temporal files](https://huggingface.co/datasets/LongLaMP/LongLaMP/tree/main/product_review_temporal) | `artifacts/product_review_temporal/raw/` |
| LongLaMP-4 | `longlamp4` | [Topic writing temporal files](https://huggingface.co/datasets/LongLaMP/LongLaMP/tree/main/topic_writing_temporal) | `artifacts/topic_writing_temporal/raw/` |

Use the LaMP **time-based split**. Save both training JSON files in `raw/train/` and both development JSON files in `raw/valid/`, keeping the filenames `dev_questions.json` and `dev_outputs.json`. For each LongLaMP task, place all `train-*.parquet` and `test-*.parquet` files directly in its `raw/` directory. Split descriptions are available on the [LaMP download page](https://lamp-benchmark.github.io/download) and the [LongLaMP dataset page](https://huggingface.co/datasets/LongLaMP/LongLaMP).

```text
artifacts/
├── LaMP_4/raw/
│   ├── train/{train_questions.json,train_outputs.json}
│   └── valid/{dev_questions.json,dev_outputs.json}
├── LaMP_5/raw/
│   ├── train/{train_questions.json,train_outputs.json}
│   └── valid/{dev_questions.json,dev_outputs.json}
├── product_review_temporal/raw/
│   ├── test-00000-of-00001.parquet
│   ├── train-00000-of-00003.parquet
│   ├── train-00001-of-00003.parquet
│   └── train-00002-of-00003.parquet
└── topic_writing_temporal/raw/
    ├── test-00000-of-00001.parquet
    ├── train-00000-of-00004.parquet
    ├── train-00001-of-00004.parquet
    ├── train-00002-of-00004.parquet
    └── train-00003-of-00004.parquet
```

## Quick start

### 1. Download all intermediate artifacts

Download all four tasks' intermediate files from [FestinaLente123/GLASS-artifacts](https://huggingface.co/datasets/FestinaLente123/GLASS-artifacts) in one command. These include prepared train/test splits, clusters and anchors, global/local vectors, and Stage1/2 outputs:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/**' --local-dir .
```

Downloaded intermediate artifacts are saved under `artifacts/` in the project root. Fixed LaMP selection files are stored with their task: `artifacts/LaMP_4/LaMP_4_users.json` and `artifacts/LaMP_4/LaMP_4_target_indices.json`, with the corresponding two files under `artifacts/LaMP_5/`.

### 2. Predict and evaluate with supplied vectors

Run Stage3 with downloaded test-cluster assignments and global/local vectors. Stage1/2 and the SAE checkpoint are not required. This example uses LaMP-4:

```bash
bash scripts/predict.sh --tasks lamp4 --gpus 0 --run runs/quick_predict
```

Predictions are saved to `runs/quick_predict/LaMP_4/generation.json`; ROUGE-1 and ROUGE-L scores are saved to `metrics.json` in the same directory. Change `--tasks` for another task.

### 3. Run each block independently

Each command uses a separate fresh output directory and reads its inputs from the downloaded artifacts. They can run independently without executing the other commands first:

| Block | Supplied inputs |
| --- | --- |
| Clustering and anchors | `train.pkl`, `test.pkl`, and reference selection metadata for LaMP |
| Global vectors | `train.pkl` and local backbone/SAE checkpoints |
| Stage1/2 and local vectors | `anchors.pkl`, `global.pt`, and backbone |
| Stage3 and evaluation | `test_clusters.pkl`, `global.pt`, `local/`, and backbone |

```bash
bash scripts/prepare_data.sh --tasks lamp4 --from cluster \
  --reference-selection --run runs/check_data
bash scripts/build_global.sh --tasks lamp4 --gpus 0 --run runs/check_global
bash scripts/build_local.sh --tasks lamp4 --gpus 0 --run runs/check_local
bash scripts/predict.sh --tasks lamp4 --gpus 0 --run runs/check_predict
```

`--reference-selection` reuses the sample selections supplied with the reference artifacts:

- **LaMP-4 and LaMP-5:** the supplied files fix the users and anchors used in this release. Anchor sampling uses one shared random stream initialized with seed 42, traversing users and clusters in order; the same inputs and order reproduce the published anchor selection. `*_users.json` fixes users during data preparation; `*_target_indices.json` fixes anchors by their zero-based positions within each history cluster.
- **LongLaMP-3 and LongLaMP-4:** anchor sampling derives an independent seed for each user and cluster, so separate selection files are not needed. Alignment of clusters, anchors, and test-cluster assignments with the reference artifacts has been verified for the first five users.

For a new experiment from raw data, all four tasks sample using the configured seed without these fixed selection files. Only the explicit `--reference-selection` option loads the LaMP selection files.

To start at Stage2 or rebuild only local directions:

```bash
bash scripts/build_local.sh --tasks lamp4 --from pairs \
  --gpus 0 --run runs/check_pairs
bash scripts/build_local.sh --tasks lamp4 --from local \
  --gpus 0 --run runs/check_local_only
```

`--from` selects the first stage, `--until` selects the last, and `--dry-run` prints the plan. See [the workflow](docs/workflow.md#starting-from-intermediate-artifacts) for each starting point's input dependencies.

## Full pipeline

After placing raw datasets and checkpoints as described under Preparation, run these four blocks in order to rebuild every intermediate result. The examples use LaMP-4; select several tasks with `--tasks lamp4 lamp5` or omit `--tasks` to run all four in order.

1. **Prepare and cluster (`prepare` → `cluster`):** organize user training histories and held-out questions, cluster the histories, sample training anchors, and assign test questions to clusters. Outputs are `train.pkl`, `test.pkl`, `clusters.json`, `anchors.pkl`, and `test_clusters.pkl`.

   ```bash
   bash scripts/prepare_data.sh --tasks lamp4
   ```

2. **Build global vectors (`global`):** teacher-force training-history responses, encode their layer-19 activations with the SAE, average the sparse codes, and decode a global direction for each user. Inputs are `train.pkl` and local backbone/SAE checkpoints; the output is `global.pt`.

   ```bash
   bash scripts/build_global.sh --tasks lamp4 --gpus 0
   ```

3. **Build local vectors (`rank` → `pairs` → `local`):** use anchors and global directions to rank demonstrations (Stage1), generate and filter contrastive pairs (Stage2), and extract local directions. Inputs are `anchors.pkl`, `global.pt`, and the backbone; outputs are `ranked.json`, `accepted.json`, `rejected.json`, and `local/`.

   ```bash
   bash scripts/build_local.sh --tasks lamp4 --gpus 0
   ```

4. **Predict and evaluate (`generate` → `evaluate`):** read test-cluster assignments and global/local vectors, apply directions to the frozen model during greedy decoding (Stage3), and calculate ROUGE-1 and ROUGE-L. Outputs are `generation.json` and `metrics.json`.

   ```bash
   bash scripts/predict.sh --tasks lamp4 --gpus 0
   ```

To execute all blocks with one command:

```bash
bash scripts/run_all.sh --tasks lamp4 --gpus 0
```

The full-pipeline outputs above go to `runs/<dataset>/`. For a separate experiment, give all four blocks, or `run_all.sh`, the same new `--run` directory.

## Parallel execution and GPU requirements

### Multiple GPUs

Model stages partition complete users across GPUs. Each GPU loads its own model, and completed shards are merged in input order. This example uses eight GPUs for Stage1, Stage2, and local vector construction:

```bash
bash scripts/build_local.sh --tasks lamp4 --gpus 0 1 2 3 4 5 6 7
```

Other model scripts accept the same `--gpus` option. For CPU clustering, `--cluster-workers 2` sets the process count. Stage logs and resolved configurations are saved inside the run directory; worker logs are saved beside stage outputs in `*.shards/` directories.

### Memory and runtime

The runtime reference hardware is **8 × NVIDIA RTX 3090 with 24 GB VRAM each**. GPUs run separate model shards, so each device must hold its model and the corresponding stage activations.

| Dataset | Runtime reference | Timing conditions |
| --- | ---: | --- |
| LaMP-4 | About 3 h | Stage1 batch=2, Stage2 batch=1 |
| LaMP-5 | About 2.5 h | Stage1 batch=2, Stage2 batch=1 |
| LongLaMP-3 / Product Review | About 28 h | Stage1/2 batch=2 |
| LongLaMP-4 / Topic Writing | About 12.5 h | Stage1/2 batch=2 |

## Citation

```bibtex
@article{chen_glass,
  title={From Profiles to Steering Vectors: Global Sparse Priors and Local Semantic Calibration for Personalized Text Generation},
  author={Chen, Liuji and Zhang, Zeyu and Zhang, Xinyuan and Nie, Shuai and Liu, Qiang and Wu, Shu and Wang, Liang}
}
```
