"""File formats and user partitions shared by the GLASS stages."""
import json
import pickle
from pathlib import Path


def _new_file(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Choose a new output path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with _new_file(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def write_pickle(path, value):
    with _new_file(path).open("xb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_vectors(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_vectors(path, value):
    import torch
    with _new_file(path).open("xb") as stream:
        torch.save(value, stream)


def shard_users(users, shard_index=0, num_shards=1):
    """Split users into ordered worker partitions."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    users = list(users)
    return users[len(users) * shard_index // num_shards:
                 len(users) * (shard_index + 1) // num_shards]


def select_users(data, config):
    users = list(data)
    requested = config.get("users")
    if requested is not None:
        requested = {str(user) for user in requested}
        users = [user for user in users if str(user) in requested]
    limit = config.get("limit_users")
    if limit is not None:
        if limit < 0:
            raise ValueError("limit_users must be nonnegative")
        users = users[:limit]
    users = shard_users(users, config.get("shard_index", 0), config.get("num_shards", 1))
    return {user: data[user] for user in users}


def user_vector(vectors, user):
    """Keep string usernames intact; accept numeric JSON/pickle key differences."""
    if user in vectors:
        return vectors[user]
    candidates = [key for key in vectors if str(key) == str(user)]
    if len(candidates) != 1:
        raise KeyError(f"No unique global vector for user {user!r}")
    return vectors[candidates[0]]
