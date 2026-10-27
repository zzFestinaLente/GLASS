"""Aggregate contrastive pair directions into local cluster directions."""

import torch


def finite_directions(directions):
    """Select pair directions available for cluster aggregation."""
    return [direction for direction in directions if torch.isfinite(direction).all().item()]


def cluster_direction(directions):
    """Compute the mean direction for a cluster, or None for an empty bank."""
    valid = finite_directions(directions)
    return torch.stack(valid).mean(0) if valid else None
