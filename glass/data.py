"""User datasets, user-local clustering, and self-supervised training anchors.

The four datasets use task-specific cluster and anchor schemas. Product Review
and Topic Writing record leave-one-out similarities for training anchors.
All paths are explicit.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import math
from pathlib import Path
import random

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from .io import load_json, load_pickle, select_users, write_json, write_pickle

DATASETS = ("LaMP_4", "LaMP_5", "product_review_temporal", "topic_writing_temporal")


def _dataset(config):
    dataset = config["dataset"]
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return dataset


def _load(path):
    return load_pickle(path) if Path(path).suffix == ".pkl" else load_json(path)


def _new_outputs(config, names):
    paths = [Path(config[name]) for name in names if config.get(name)]
    if len(set(paths)) != len(paths):
        raise ValueError("Each output must have a distinct path")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Choose a new output path: {path}")


def _user_config(config):
    result = dict(config)
    if isinstance(result.get("users"), (str, Path)):
        result["users"] = load_json(result["users"])
    return result


def _select(data, config):
    return select_users(data, _user_config(config))


def run_prepare(config):
    """Prepare train/test user dictionaries from parquet or LaMP JSON.

    Required: dataset, raw_dir, train_output, test_output. Optional: users (a
    list or JSON path), threshold=200, seed=42, limit_users, shard_index,
    num_shards. Supply users to select a specific user set; otherwise seed controls
    selection from the eligible users. For LaMP, raw_dir contains train and valid
    subdirectories with question and output JSON files.
    """
    dataset = _dataset(config)
    _new_outputs(config, ("train_output", "test_output"))
    raw = Path(config["raw_dir"])
    train, test = defaultdict(list), defaultdict(list)
    activity = {}
    lamp = dataset.startswith("LaMP_")
    if lamp:
        for target, questions, answers in (
            (train, raw / "train/train_questions.json", raw / "train/train_outputs.json"),
            (test, raw / "valid/dev_questions.json", raw / "valid/dev_outputs.json"),
        ):
            questions, answers = load_json(questions), load_json(answers)["golds"]
            if len(questions) != len(answers):
                raise ValueError("Question and gold lengths differ")
            for sample, answer in zip(questions, answers):
                if sample["id"] != answer["id"]:
                    raise ValueError("Question and gold IDs differ")
                uid = sample["user_id"]
                target[uid].append({"id": sample["id"], "input": sample["input"],
                    "profile": sample["profile"], "user_id": uid, "output": answer["output"]})
                if target is train:
                    activity[uid] = len(sample["profile"])
    else:
        import pandas as pd
        next_id = 0
        for target, split in ((train, "train"), (test, "test")):
            paths = sorted(p for p in raw.glob("*.parquet") if split in p.name)
            if not paths:
                raise FileNotFoundError(f"No {split} parquet files in {raw}")
            for path in paths:
                for sample in pd.read_parquet(path).to_dict(orient="records"):
                    uid = sample.get("name", sample.get("author", sample.get("reviewerId")))
                    profile = sample["profile"]
                    profile = profile.tolist() if hasattr(profile, "tolist") else profile
                    if target is train:
                        row = {"input": sample["input"], "user_id": uid, "output": sample["output"],
                               "profile": profile, "id": next_id}
                    else:
                        row = {"input": sample["input"], "profile": profile, "user_id": uid,
                               "output": sample["output"], "id": next_id}
                    next_id += 1
                    for history in profile:
                        history["id"] = next_id
                        next_id += 1
                    target[uid].append(row)
    eligible = set(train) & set(test)
    if lamp:
        eligible = {uid for uid in eligible if activity[uid] > 10}
    requested = _user_config(config).get("users")
    if requested is None:
        population = sorted(eligible)
        threshold = config.get("threshold", 200)
        if threshold > len(population):
            raise ValueError(f"Requested {threshold} users from {len(population)} eligible users")
        selected = set(random.Random(config.get("seed", 42)).sample(population, threshold))
    else:
        requested = {str(uid) for uid in requested}
        selected = {uid for uid in eligible if str(uid) in requested}
        if {str(uid) for uid in selected} != requested:
            raise ValueError("Some requested users are absent or fail the profile-size constraint")
    if lamp:
        train = {uid: rows for uid, rows in train.items() if uid in selected}
        test = {uid: rows for uid, rows in test.items() if uid in selected}
    else:
        train = {uid: train[uid] for uid in sorted(train) if uid in selected}
        test = {uid: test[uid] for uid in sorted(test) if uid in selected}
    train = _select(train, config)
    selected_users = {str(uid) for uid in train}
    test = {uid: rows for uid, rows in test.items() if str(uid) in selected_users}
    write_pickle(config["train_output"], train)
    write_pickle(config["test_output"], test)
    return {"dataset": dataset, "train_output": str(config["train_output"]),
            "test_output": str(config["test_output"]), "train_users": len(train),
            "test_users": len(test), "train_samples": sum(map(len, train.values())),
            "test_samples": sum(map(len, test.values())),
            "user_selection": "fixed" if requested is not None else "seeded_sorted_population"}


def fit_tfidf(texts, max_features=2048):
    """Fit word/bigram TF-IDF, relaxing pruning if no terms remain."""
    count = len(texts)
    kwargs = dict(max_features=max_features, ngram_range=(1, 2),
                  min_df=2 if count >= 10 else 1, max_df=0.9 if count >= 10 else 1.0,
                  sublinear_tf=True)
    try:
        vectorizer = TfidfVectorizer(**kwargs)
        return vectorizer, vectorizer.fit_transform(texts)
    except ValueError as error:
        if "After pruning, no terms remain" not in str(error):
            raise
        kwargs.update(min_df=1, max_df=1.0)
        vectorizer = TfidfVectorizer(**kwargs)
        return vectorizer, vectorizer.fit_transform(texts)


def choose_best_k(matrix, upper, min_cluster_size=6, random_state=42):
    best = None
    for count in range(2, upper + 1):
        labels = KMeans(n_clusters=count, random_state=random_state, n_init=20).fit_predict(matrix)
        if min(Counter(labels).values()) < min_cluster_size:
            continue
        score = silhouette_score(matrix, labels, metric="cosine")
        if best is None or score > best[0]:
            best = (score, count, labels)
    return (1, [0] * matrix.shape[0]) if best is None else (best[1], best[2])


def _training_history(sample, dataset):
    if dataset == "LaMP_4":
        return {"text": sample["input"].replace("Generate a headline for the following article: ", ""),
                "title": sample["output"], "id": sample["id"]}
    if dataset == "LaMP_5":
        return {"abstract": sample["input"].replace("Generate a title for the following abstract of a paper: ", ""),
                "title": sample["output"], "id": sample["id"]}
    if dataset == "topic_writing_temporal":
        return {"summary": sample["input"].replace("Generate the content for a reddit post ", ""),
                "content": sample["output"], "id": sample["id"]}
    return {"input": sample["input"], "reviewText": sample["output"], "id": sample["id"]}


def _question(entry, dataset):
    if dataset == "product_review_temporal":
        if "input" in entry:
            return entry["input"].replace(
                "Generate the review text written by a reviewer who has a given an overall rating of", ""
            ).replace("for a product with description", "description").replace(
                "The summary of the review text is", "summary")
        return (f'"{entry["overall"]}"' f'description "{entry["description"]}". '
                f'summary "{entry["summary"]}".')
    key = {"LaMP_4": "text", "LaMP_5": "abstract", "topic_writing_temporal": "summary"}[dataset]
    return str(entry[key]).strip() if entry.get(key) else None


def _profiles(data, dataset):
    profiles, training = {}, defaultdict(list)
    for uid, samples in data.items():
        for sample in samples:
            user = sample.get("user_id", uid) or sample.get("name")
            if user is None:
                continue
            training[user].append(_training_history(sample, dataset))
            profile = sample.get("profile") or sample.get("history")
            if profile is not None and user not in profiles:
                profiles[user] = copy.deepcopy(profile)
    for uid, samples in training.items():
        base = profiles.get(uid, [])
        if not isinstance(base, list):
            base = [base]
        profiles[uid] = base + samples
    return profiles


def _cluster_user(job):
    uid, entries, config = job
    with threadpool_limits(limits=1):
        questions = [{"index": index, "question": _question(entry, config["dataset"]), "raw": entry}
                     for index, entry in enumerate(entries) if isinstance(entry, dict)]
        questions = [row for row in questions if row["question"]]
        if not questions:
            return str(uid), {"profile_size": len(entries), "used_question_count": 0, "clusters": []}
        texts = [row["question"] for row in questions]
        if len(set(texts)) == 1:
            best_k, labels = 1, [0] * len(texts)
        else:
            _, matrix = fit_tfidf(texts, config.get("max_features", 2048))
            upper = min(config.get("k", 100), int(math.sqrt(len(texts))) + 1, len(texts))
            best_k, labels = choose_best_k(matrix, upper, config.get("choose_k_min_cluster_size", 6),
                                           config.get("kmeans_random_state", 42))
        clusters = []
        for label in sorted(set(labels)):
            members = [questions[index] for index, value in enumerate(labels) if value == label]
            clusters.append({"cluster_id": int(label), "size": len(members), "questions": members})
        return str(uid), {"profile_size": len(entries), "used_question_count": len(questions),
                          "best_k": int(best_k), "clusters": clusters}


def _ordered_map(function, jobs, workers):
    if workers < 1:
        raise ValueError("num_workers must be positive")
    if workers == 1:
        return list(map(function, jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs))


def _target_choices(config):
    choices = config.get("target_indices", {})
    if isinstance(choices, (str, Path)):
        choices = load_json(choices)
    if isinstance(choices, list):
        return {(str(row["uid"]), int(row["cluster_id"])): row["indices"] for row in choices}
    return {(str(uid), int(cid)): indices for uid, values in choices.items() for cid, indices in values.items()}


def _stable_uid(uid):
    try:
        return int(uid)
    except ValueError:
        return int.from_bytes(str(uid).encode("utf-8"), "little") % (2 ** 31 - 1)


def _anchor(raw, dataset):
    row = copy.deepcopy(raw)
    if dataset in ("LaMP_4", "LaMP_5"):
        key = "text" if dataset == "LaMP_4" else "abstract"
        task = ("Generate a headline for the following article: " if dataset == "LaMP_4"
                else "Generate a title for the following abstract of a paper: ")
        row["input"] = task + row[key]
        row["output"] = row["title"]
        del row[key], row["title"]
    elif dataset == "topic_writing_temporal":
        row["input"] = f"Generate the content for a reddit post {row['summary']}"
        row["output"] = row["content"]
        del row["summary"], row["content"]
    else:
        if "input" not in row:
            row["input"] = (
                f'Generate the review text written by a reviewer who has a given an overall rating of "{row["overall"]}" '
                f'for a product with description "{row["description"]}". '
                f'The summary of the review text is "{row["summary"]}".')
            del row["overall"], row["description"], row["summary"]
        row["output"] = row["reviewText"]
        del row["reviewText"]
    return row


def _centers(clusters, max_features):
    texts = [m["question"] for cluster in clusters for m in cluster["questions"] if m.get("question")]
    if not texts:
        return None
    vectorizer, _ = fit_tfidf(texts, max_features)
    ids, centers, sums, sizes, lookup = [], [], [], [], []
    for cluster in clusters:
        members = [m["question"] for m in cluster["questions"] if m.get("question")]
        if not members:
            continue
        matrix = vectorizer.transform(members)
        ids.append(int(cluster["cluster_id"]))
        centers.append(np.asarray(matrix.mean(axis=0)).ravel())
        sums.append(np.asarray(matrix.sum(axis=0)).ravel())
        sizes.append(matrix.shape[0])
        lookup.append(cluster)
    return vectorizer, ids, np.vstack(centers), sums, sizes, lookup


def _self_supervised(clusters, config):
    dataset = config["dataset"]
    lamp = dataset.startswith("LaMP_")
    limit = config.get("max_samples_limit", 5 if lamp else 10)
    minimum = config.get("strict_min_cluster_size", 6 if lamp else 4)
    if config.get("strict_stop", True):
        for uid, info in clusters.items():
            for cluster in info["clusters"]:
                if len(cluster["questions"]) < minimum:
                    raise ValueError(f"User {uid}, cluster {cluster['cluster_id']} has fewer than {minimum} members")
    choices = _target_choices(config)
    target_ids = config.get("target_ids")
    if isinstance(target_ids, (str, Path)):
        target_ids = load_json(target_ids)
    rng = random.Random(config.get("seed", 42))
    result = defaultdict(list)
    for uid, info in clusters.items():
        centers = None if lamp else _centers(info["clusters"], config.get("max_features", 2048))
        for cluster in info["clusters"]:
            cid, members = int(cluster["cluster_id"]), cluster["questions"]
            count = min(int(len(members) / 2), limit)
            if (str(uid), cid) in choices:
                selected = list(choices[str(uid), cid])
            elif target_ids is not None:
                selected_ids = target_ids.get(str(uid), []) if isinstance(target_ids, dict) else target_ids
                index = {str(member["raw"]["id"]): i for i, member in enumerate(members)}
                selected = [index[str(value)] for value in selected_ids if str(value) in index]
            elif lamp:
                selected = rng.sample(range(len(members)), count)
            else:
                seed = (config.get("seed", 42) * 1000003 + _stable_uid(uid) * 9176 + cid * 1009) % (2 ** 32)
                selected = random.Random(seed).sample(range(len(members)), count)
            if len(set(selected)) != len(selected) or any(i < 0 or i >= len(members) for i in selected):
                raise ValueError(f"Invalid target indices for {uid}/{cid}")
            for index in selected:
                target = members[index]
                row = _anchor(target["raw"], dataset)
                row["profile"] = [member["raw"] for i, member in enumerate(members) if i != index]
                row["profile_cluster_id"] = cid
                row.setdefault("user_id", int(uid) if lamp else uid)
                if not lamp:
                    ranking = []
                    if config.get("add_cluster_ranking", True) and target.get("question") and centers is not None:
                        vectorizer, ids, matrix, sums, sizes, _ = centers
                        query = vectorizer.transform([target["question"]])
                        full = cosine_similarity(query, matrix)[0]
                        used = full.copy()
                        own = ids.index(cid) if cid in ids else None
                        loo = None
                        if own is not None and sizes[own] >= 2:
                            center = (sums[own] - query.toarray().ravel()) / float(sizes[own] - 1)
                            loo = float(cosine_similarity(query, center.reshape(1, -1))[0, 0])
                            used[own] = loo
                        order = np.argsort(-used)
                        top = config.get("rank_topk")
                        if isinstance(top, int) and top > 0:
                            order = order[:min(top, len(order))]
                        for rank, j in enumerate(order):
                            j = int(j)
                            payload = {"rank": rank + 1, "cluster_id": int(ids[j]),
                                "similarity": float(used[j]), "distance": float(1.0 - used[j]), "size": int(sizes[j])}
                            if j == own:
                                payload.update(similarity_full=float(full[j]), distance_full=float(1.0 - full[j]),
                                    similarity_loo=loo, distance_loo=float(1.0 - loo) if loo is not None else None,
                                    ranking_mode="own_cluster_loo_others_full")
                            ranking.append(payload)
                    row["profile_cluster_ranking"] = ranking
                result[int(uid) if lamp else uid].append(row)
    return result


def _assign_user(job):
    uid, samples, clusters, config = job
    lamp = config["dataset"].startswith("LaMP_")
    with threadpool_limits(limits=1):
        centers = _centers(clusters, config.get("max_features", 2048))
        output = []
        for sample in samples:
            row = copy.deepcopy(sample)
            text = str(row.get("input", "")).strip()
            if ":" in text:
                rest = text.split(":", 1)[1].strip()
                if rest:
                    text = rest
            assigned, ranking = None, []
            if text and centers is not None:
                vectorizer, ids, matrix, _, sizes, lookup = centers
                sims = cosine_similarity(vectorizer.transform([text]), matrix)[0]
                order = np.argsort(-sims)
                top = config.get("rank_topk")
                if isinstance(top, int) and top > 0:
                    order = order[:min(top, len(order))]
                for rank, j in enumerate(order):
                    j = int(j)
                    payload = {"rank": rank + 1, "cluster_id": int(ids[j]), "similarity": float(sims[j])}
                    if not lamp:
                        payload["distance"] = float(1.0 - sims[j])
                    payload["size"] = int(lookup[j].get("size", sizes[j])) if lamp else int(sizes[j])
                    ranking.append(payload)
                if len(order):
                    assigned = lookup[int(order[0])]
            if assigned:
                key = "profile" if "profile" in row else "history" if "history" in row else "profile"
                row[key] = [member["raw"] for member in assigned["questions"]]
                row["profile_cluster_id"] = int(assigned["cluster_id"])
            else:
                row["profile_cluster_id"] = None
            if config.get("add_cluster_ranking", True):
                row["profile_cluster_ranking"] = ranking
            output.append(row)
        return uid if lamp else str(uid), output


def run_cluster(config):
    """Cluster histories, construct anchors, and optionally assign test queries.

    Required: dataset and train_input (unless clusters_input is supplied).
    Explicit outputs: clusters_output, train_output, test_output; omit unneeded
    outputs. test_input is needed only for test_output. Existing clusters_input
    and target_indices JSON let individual stages use fixed inputs.
    test_clusters_input can independently fix the test cluster map.
    num_workers parallelizes independent users and preserves insertion order.
    """
    _dataset(config)
    _new_outputs(config, ("clusters_output", "train_output", "test_output"))
    workers = config.get("num_workers", 1)
    with threadpool_limits(limits=1):
        if config.get("clusters_input"):
            clusters = load_json(config["clusters_input"])
            selected_clusters = _select(clusters, config)
        else:
            train = _select(_load(config["train_input"]), config)
            profiles = _profiles(train, config["dataset"])
            jobs = [(uid, entries, dict(config)) for uid, entries in profiles.items()]
            clusters = dict(_ordered_map(_cluster_user, jobs, workers))
            selected_clusters = clusters
        if config.get("clusters_output"):
            write_json(config["clusters_output"], selected_clusters)
        training = _self_supervised(selected_clusters, config) if config.get("train_output") else None
        if training is not None:
            write_pickle(config["train_output"], training)
        assigned = None
        if config.get("test_output"):
            selected_users = {str(uid) for uid in selected_clusters}
            test = {uid: samples for uid, samples in _load(config["test_input"]).items()
                    if str(uid) in selected_users}
            test_clusters = load_json(config["test_clusters_input"]) if config.get("test_clusters_input") else clusters
            jobs = [(uid, samples, test_clusters.get(str(uid), {}).get("clusters", []), dict(config))
                    for uid, samples in test.items()]
            assigned = dict(_ordered_map(_assign_user, jobs, workers))
            write_pickle(config["test_output"], assigned)
    return {"dataset": config["dataset"], "cluster_users": len(selected_clusters),
            "clusters": sum(len(info["clusters"]) for info in selected_clusters.values()),
            "train_samples": None if training is None else sum(map(len, training.values())),
            "test_samples": None if assigned is None else sum(map(len, assigned.values())),
            "clusters_source": "provided" if config.get("clusters_input") else "computed",
            "target_selection": "fixed" if config.get("target_indices") or config.get("target_ids") else "seeded",
            "num_workers": workers, "numpy": np.__version__}
