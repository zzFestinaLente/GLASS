# Running the pipeline

[English](workflow.md) | [简体中文](workflow_zh.md)

All stages use the environment installed from `requirements.txt`. Preparation and clustering run on CPU; vector construction and prediction use the selected GPUs. Configure local backbone and SAE checkpoints in [configs/paths.json](../configs/paths.json).

## Four script blocks

| Script | Stages | Main outputs |
| --- | --- | --- |
| `scripts/prepare_data.sh` | `prepare` → `cluster` | `train.pkl`, `test.pkl`, `clusters.json`, `anchors.pkl`, `test_clusters.pkl` |
| `scripts/build_global.sh` | `global` | `global.pt` |
| `scripts/build_local.sh` | `rank` → `pairs` → `local` | `ranked.json`, `accepted.json`, `rejected.json`, `local/` |
| `scripts/predict.sh` | `generate` → `evaluate` | `generation.json`, `metrics.json` |
| `scripts/run_all.sh` | All stages above | Complete task outputs |

Run the blocks in order from the repository root:

```bash
bash scripts/prepare_data.sh --tasks lamp4
bash scripts/build_global.sh --tasks lamp4 --gpus 0
bash scripts/build_local.sh --tasks lamp4 --gpus 0
bash scripts/predict.sh --tasks lamp4 --gpus 0
```

Or execute the same full sequence with one command:

```bash
bash scripts/run_all.sh --tasks lamp4 --gpus 0
```

Use `--tasks lamp4 lamp5` to select several tasks, or omit `--tasks` to run all four tasks in order. Each task completes the selected stages before the next task starts.

| Task name | Configuration | Data/output subdirectory |
| --- | --- | --- |
| `lamp4` | `configs/lamp4.json` | `LaMP_4` |
| `lamp5` | `configs/lamp5.json` | `LaMP_5` |
| `longlamp3` | `configs/longlamp3.json` | `product_review_temporal` |
| `longlamp4` | `configs/longlamp4.json` | `topic_writing_temporal` |

Scripts use local inputs and checkpoints and perform no downloads. Download inputs explicitly from [FestinaLente123/GLASS-artifacts](https://huggingface.co/datasets/FestinaLente123/GLASS-artifacts) before selecting a starting point.

## Inputs and outputs

For each stage input, the scripts first look in `runs/<dataset>/`, then in `artifacts/<dataset>/`. Newly generated outputs are always written to `runs/<dataset>/`. Downloaded reference files remain in `artifacts/`.

Prepared test files can be named `test.pkl` or `seen_test.pkl`. The search order is `runs/<dataset>/test.pkl`, `runs/<dataset>/seen_test.pkl`, `artifacts/<dataset>/test.pkl`, then `artifacts/<dataset>/seen_test.pkl`. Other stage inputs use their configured filenames.

User and anchor sampling use the configured seed, 42 by default. The raw-data pipeline does not require selection metadata. To reuse the supplied LaMP user and anchor selections, add `--reference-selection` to `prepare_data.sh` or `run_all.sh`. This option reads `artifacts/<dataset>/<dataset>_users.json` during preparation and `artifacts/<dataset>/<dataset>_target_indices.json` during clustering. Only the selected stages require their corresponding selection files.

Before a run starts, the scripts check that all selected outputs are new. Existing output files or shard directories stop the run before computation; completed stages are not skipped automatically. Select a later starting stage with `--from` when continuing a chain.

## Starting from intermediate artifacts

Each block accepts `--from STAGE` and `--until STAGE`, inclusive. The stage must belong to that block; `run_all.sh` accepts any stage in the full sequence. For example, `--from pairs` on `build_local.sh` executes `pairs` followed by `local`, while `--from pairs --until pairs` executes Stage2 alone.

The table lists the inputs needed to run from each starting stage through the end of its block. Download intermediate inputs into `artifacts/<dataset>/`; evaluation uses `runs/<dataset>/generation.json` produced by local prediction. Other inputs already present in `runs/<dataset>/` do not need downloading again.

| Start | Script | Required inputs (in `artifacts/<dataset>/` unless otherwise specified) |
| --- | --- | --- |
| `cluster` | `prepare_data.sh --from cluster` | `train.pkl`, `test.pkl` |
| `global` | `build_global.sh` | `train.pkl` |
| `rank` | `build_local.sh --from rank` | `anchors.pkl`, `global.pt` |
| `pairs` | `build_local.sh --from pairs` | `anchors.pkl`, `ranked.json`, `global.pt` |
| `local` | `build_local.sh --from local` | `accepted.json`, `global.pt` |
| `generate` | `predict.sh --from generate` | `test_clusters.pkl`, `global.pt`, `local/` |
| `evaluate` | `predict.sh --from evaluate` | Locally generated `runs/<dataset>/generation.json` |

The global-vector block needs `train.pkl` and the backbone/SAE checkpoints; it does not require anchors. The local-vector block starts from `anchors.pkl` and saved global directions. Other model stages use the backbone and saved directions without loading the SAE checkpoint. To continue across blocks with `run_all.sh --from ...`, also provide the later blocks' inputs, such as `test_clusters.pkl` for prediction.

### Begin with prepared user splits

Download prepared splits, then run clustering with seeded anchor sampling:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/train.pkl' \
            'artifacts/LaMP_4/test.pkl' \
  --local-dir .
bash scripts/prepare_data.sh --tasks lamp4 --from cluster
```

Continue with `build_global.sh`, `build_local.sh`, and `predict.sh`. This route skips raw-data preparation while rebuilding every downstream intermediate result.

To reuse the supplied anchor selection instead, also download `artifacts/LaMP_4/LaMP_4_target_indices.json` and add `--reference-selection` to the clustering command.

### Begin with Stage1

Download the anchors and global bank, then rebuild rankings, pairs, and local directions:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/anchors.pkl' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from rank --gpus 0
```

### Begin with Stage2

Reuse the supplied rankings while regenerating contrastive pairs and local directions:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/anchors.pkl' \
            'artifacts/LaMP_4/ranked.json' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from pairs --gpus 0
```

### Begin with local direction construction

Download accepted Stage2 pairs and the global bank:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/accepted.json' \
            'artifacts/LaMP_4/global.pt' \
  --local-dir .
bash scripts/build_local.sh --tasks lamp4 --from local --gpus 0
```

### Begin with prediction

Use supplied vectors and test-cluster assignments to run Stage3 and scoring:

```bash
hf download FestinaLente123/GLASS-artifacts --repo-type dataset \
  --include 'artifacts/LaMP_4/test_clusters.pkl' \
            'artifacts/LaMP_4/global.pt' \
            'artifacts/LaMP_4/local/*' \
  --local-dir .
bash scripts/predict.sh --tasks lamp4 --gpus 0
```

If `runs/LaMP_4/generation.json` has already been generated locally, rerun scoring without repeating model inference:

```bash
bash scripts/predict.sh --tasks lamp4 --from evaluate
```

For another task, replace the task name and artifact subdirectory using the mapping above.

## Progress, logs, and planning

Stage labels identify the current task and operation. Interactive terminals display progress bars; redirected logs receive stage start/end messages and periodic updates. A stage is complete after its process exits successfully, including output writing and shard merging.

Each launch saves stage logs and resolved configurations under `runs/pipeline_<timestamp>/`. Model-worker logs are also saved beside stage outputs in `*.shards/` directories. A failed stage stops the pipeline and reports its log. Ctrl-C stops the active stage and its workers.

Use `--dry-run` to print selected stages, resolved inputs, and outputs without loading models or starting computation:

```bash
bash scripts/build_local.sh --tasks lamp4 --from pairs --gpus 0 1 --dry-run
```

## Parallel execution and parameters

Pass multiple GPU indices to run whole-user shards:

```bash
bash scripts/build_local.sh --tasks lamp4 --gpus 0 1 2 3 4 5 6 7
```

Each worker loads one model and receives complete users; the parent merges outputs in input order after all workers succeed. Stage2 workers retain access to the full other-user negative pool. Whole-user sharding does not increase the configured batch size automatically.

Task parameters live in `configs/<task>.json`. Each stage reads its own section together with `common`; edit the task configuration to change parameters such as batch sizes or steering strengths. For a single stage, the Python CLI also accepts `--set KEY=VALUE`, including JSON numbers, lists, booleans, and null:

```bash
python -m glass rank --config configs/lamp4.json --gpus 0 1 \
  --set input_path=artifacts/LaMP_4/anchors.pkl \
  --set global_vectors_path=artifacts/LaMP_4/global.pt \
  --set output_path=runs/LaMP_4/ranked.json
```

## Evaluation

ROUGE-1 and ROUGE-L use `rouge-score` with stemming. Aggregates are arithmetic means of per-example F1, precision, and recall on the 0–1 scale. Empty generation strings receive zero ROUGE.

METEOR is optional (`--set meteor=true` on the single-stage evaluation command) and uses NLTK tokenization and WordNet. Install its resources before an offline run:

```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
```

Generation and evaluation do not download these resources implicitly.
