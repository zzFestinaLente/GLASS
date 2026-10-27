#!/usr/bin/env python3
"""Run grouped GLASS stages, optionally resuming from downloaded artifacts."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from pipeline_runtime import TASKS, dump_new, run_stage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glass.paths import expand_setting, load_paths


BLOCKS = {
    "prepare": ("prepare", "cluster"),
    "global": ("global",),
    "local": ("rank", "pairs", "local"),
    "predict": ("generate", "evaluate"),
    "all": ("prepare", "cluster", "global", "rank", "pairs", "local", "generate", "evaluate"),
}
MODEL_STAGES = {"global", "rank", "pairs", "local", "generate"}
OUTPUT_KEYS = {
    "prepare": ("train_output", "test_output"),
    "cluster": ("clusters_output", "train_output", "test_output"),
    "global": ("output_path",), "rank": ("output_path",),
    "pairs": ("output_path", "error_output_path"), "local": ("output_path",),
    "generate": ("output_path",), "evaluate": ("output_path",),
}


def path_present(path):
    return path.exists() or path.is_symlink()


def output_paths(stage, config):
    outputs = [Path(config[key]) for key in OUTPUT_KEYS[stage] if config.get(key)]
    if stage in MODEL_STAGES:
        output = Path(config["output_path"])
        outputs.append(output.parent / (output.name + ".shards"))
    return outputs


def require_path(path, *, directory=False):
    path = Path(path)
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"Required {'directory' if directory else 'file'}: {path}")


def resolve_config(document, stage, paths, *, planned=(), dry_run=False,
                   cluster_workers=None, reference_selection=False):
    """Prefer this run's upstream results, then the downloaded artifact bank."""
    config = {key: expand_setting(value, paths)
              for key, value in {**document.get("common", {}), **document[stage]}.items()}
    dataset = config["dataset"]
    roots = [Path(paths[name]) / dataset for name in ("GLASS_RUN", "GLASS_DATA")]
    if reference_selection and dataset.startswith("LaMP_"):
        selection = {"prepare": ("users", "users"), "cluster": ("target_indices", "target_indices")}
        if stage in selection:
            key, suffix = selection[stage]
            config[key] = str(Path(paths["GLASS_DATA"]) / dataset / f"{dataset}_{suffix}.json")

    def source(*names, directory=False):
        candidates = [root / name for root in roots for name in names]
        for path in candidates:
            if path in planned or (path.is_dir() if directory else path.is_file()):
                return str(path)
        if not dry_run:
            raise FileNotFoundError("Missing stage input; looked for: " + ", ".join(map(str, candidates)))
        return str(roots[1] / names[0])

    if stage == "prepare":
        if not dry_run:
            require_path(config["raw_dir"], directory=True)
            if isinstance(config.get("users"), str):
                require_path(config["users"])
    elif stage == "cluster":
        config["train_input"] = source("train.pkl")
        config["test_input"] = source("test.pkl", "seen_test.pkl")
        if cluster_workers is not None:
            config["num_workers"] = cluster_workers
        if not dry_run and isinstance(config.get("target_indices"), str):
            require_path(config["target_indices"])
    elif stage == "global":
        config["input_path"] = source("train.pkl")
    elif stage in ("rank", "pairs"):
        config["input_path"] = source("anchors.pkl")
        config["global_vectors_path"] = source("global.pt")
        if stage == "pairs":
            config["ranked_path"] = source("ranked.json")
    elif stage == "local":
        config["input_path"] = source("accepted.json")
        config["global_vectors_path"] = source("global.pt")
    elif stage == "generate":
        config["input_path"] = source("test_clusters.pkl")
        if config["global_strength"] != 0:
            config["global_vectors_path"] = source("global.pt")
        if config["local_strength"] != 0:
            config["local_vectors_path"] = source("local", directory=True)
    elif stage == "evaluate":
        config["input_path"] = source("generation.json")
    if not dry_run:
        if stage in MODEL_STAGES:
            require_path(config["model_path"], directory=True)
        if stage == "global":
            require_path(config["sae_model_path"], directory=True)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=BLOCKS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpus", nargs="+", default=["0"], help="GPU numbers or UUIDs for whole-user shards")
    parser.add_argument("--from", dest="from_stage", help="Begin at this stage within the selected block")
    parser.add_argument("--until", help="Stop after this stage within the selected block")
    parser.add_argument("--data", type=Path, help="Input artifact root (default: project/artifacts)")
    parser.add_argument("--run", type=Path, help="New output root (default: project/runs)")
    parser.add_argument("--cluster-workers", type=int, help="Override clustering CPU worker count")
    parser.add_argument("--reference-selection", action="store_true",
                        help="Reuse supplied LaMP user and anchor selections during preparation/clustering")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and commands without writing or running stages")
    args = parser.parse_args()
    stages = BLOCKS[args.block]
    for value in (args.from_stage, args.until):
        if value is not None and value not in stages:
            parser.error(f"Stage {value!r} is outside {args.block}: {', '.join(stages)}")
    start = stages.index(args.from_stage) if args.from_stage else 0
    end = stages.index(args.until) + 1 if args.until else len(stages)
    if start >= end:
        parser.error("--from must not come after --until")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.gpus)) != len(args.gpus):
        parser.error("Select each task and each GPU once")
    if args.cluster_workers is not None and args.cluster_workers < 1:
        parser.error("--cluster-workers must be positive")
    return args, stages[start:end]


def main():
    args, stages = parse_args()
    paths = load_paths(ROOT)
    for key, value in (("GLASS_DATA", args.data), ("GLASS_RUN", args.run)):
        if value is not None:
            paths[key] = str(value.expanduser().resolve())
    logdir = Path(paths["GLASS_RUN"]) / ("pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    documents = {task: json.loads((ROOT / "configs" / (task + ".json")).read_text())
                 for task in args.tasks}
    plans, claimed = [], set()
    # Validate the complete selected chain before starting any expensive stage.
    for task in args.tasks:
        planned = set()
        for stage in stages:
            config = resolve_config(documents[task], stage, paths, planned=planned,
                                    dry_run=args.dry_run, cluster_workers=args.cluster_workers,
                                    reference_selection=args.reference_selection)
            outputs = output_paths(stage, config)
            for path in outputs:
                if path in claimed:
                    raise ValueError(f"Repeated output target: {path}")
                if not args.dry_run and path_present(path):
                    raise FileExistsError(f"Choose a fresh --run directory or start after existing outputs: {path}")
                claimed.add(path)
            planned.update(outputs)
            config_path = logdir / f"{task}_{stage}_config.json"
            command = [sys.executable, "-m", "glass", stage, "--config", str(config_path)]
            if stage in MODEL_STAGES:
                command += ["--gpus", *args.gpus]
            plans.append((task, stage, config_path, command, config))
    if args.dry_run:
        for task, stage, _, command, config in plans:
            print(json.dumps({"task": task, "stage": stage, "paths_resolved": True,
                              "config": config, "command": command}, ensure_ascii=False, indent=2))
        return
    logdir.mkdir(parents=True, exist_ok=False)
    dump_new(logdir / "settings.json", {"block": args.block, "stages": stages,
             "tasks": args.tasks, "gpus": args.gpus, "paths": paths,
             "reference_selection": args.reference_selection})
    env = dict(os.environ, **paths, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for task, stage, config_path, command, _ in plans:
        # Earlier selected stages now exist: select their actual files at launch.
        config = resolve_config(documents[task], stage, paths, cluster_workers=args.cluster_workers,
                                reference_selection=args.reference_selection)
        for output in output_paths(stage, config):
            if path_present(output):
                raise FileExistsError(f"Output appeared after preflight: {output}")
        dump_new(config_path, {"paths_resolved": True, stage: config})
        dump_new(logdir / f"{task}_{stage}_command.json", command)
        print(shlex.join(command), flush=True)
        label = f"[{task} {stages.index(stage) + 1}/{len(stages)} {stage}]"
        run_stage(command, logdir / f"{task}_{stage}.log", cwd=ROOT, env=env,
                  label=label, stage=stage)
    print(f"Completed {args.block}; settings and logs: {logdir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)
