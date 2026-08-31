"""Training-only supervision helpers for immutable M75-C graph records."""

import torch
import torch.nn.functional as functional


def supervision_from_graph(graph, device):
    """Derive targets outside the M75 model input boundary."""
    labels = graph['node_labels'].to(device=device, dtype=torch.float32)
    target_ids = graph['node_target_ids'].to(device=device, dtype=torch.long)
    bins = graph['node_time_bins'].to(device=device, dtype=torch.long)
    window_ids = torch.full_like(target_ids, -1)
    next_window = 0
    for target_id, time_bin in torch.unique(
        torch.stack((target_ids[labels > 0.5], bins[labels > 0.5]), dim=1),
        dim=0,
    ) if torch.any(labels > 0.5) else []:
        mask = (labels > 0.5) & (target_ids == target_id) & (bins == time_bin)
        window_ids[mask] = next_window
        next_window += 1
    edge_index = graph['edge_index'].to(device=device)
    if edge_index.numel():
        source, destination = edge_index
        edge_labels = (
            (labels[source] > 0.5)
            & (labels[destination] > 0.5)
            & (target_ids[source] == target_ids[destination])
            & (target_ids[source] > 0)
        ).to(dtype=torch.float32)
    else:
        edge_labels = labels.new_empty((0,))
    target_bins = bins[(labels > 0.5) & (target_ids > 0)].unique()
    no_target_bin = torch.ones_like(labels, dtype=torch.bool)
    for time_bin in target_bins:
        no_target_bin &= bins != time_bin
    return labels, window_ids, edge_labels, no_target_bin


def m75c_loss(graph, node_logits, edge_logits, device):
    """Fixed M75-C node, coverage, edge, and false-budget losses."""
    labels, window_ids, edge_labels, no_target_bin = supervision_from_graph(graph, device)
    positive = labels > 0.5
    negative = ~positive
    positive_weight = min(32.0, float(negative.sum()) / max(float(positive.sum()), 1.0))
    node_loss = functional.binary_cross_entropy_with_logits(
        node_logits,
        labels,
        pos_weight=torch.tensor(positive_weight, device=device),
    )
    coverage_terms = []
    for window_id in window_ids[window_ids >= 0].unique(sorted=True):
        probability = torch.sigmoid(node_logits[window_ids == window_id])
        coverage_terms.append(-torch.log1p(-probability.clamp(max=1.0 - 1e-6)).sum().neg().expm1().clamp_min(1e-6))
    coverage_loss = torch.stack(coverage_terms).mean() if coverage_terms else node_logits.new_zeros(())
    positive_edges = edge_labels > 0.5
    if edge_logits.numel() and positive_edges.any():
        negative_edges = torch.nonzero(~positive_edges, as_tuple=False).reshape(-1)
        selected = torch.cat((
            torch.nonzero(positive_edges, as_tuple=False).reshape(-1),
            negative_edges[:min(negative_edges.numel(), int(positive_edges.sum()) * 4)],
        ))
        edge_loss = functional.binary_cross_entropy_with_logits(edge_logits[selected], edge_labels[selected])
    else:
        edge_loss = node_logits.new_zeros(())
    bins = graph['node_time_bins'].to(device=device)
    budget_terms = []
    for time_bin in bins[no_target_bin].unique(sorted=True):
        values = node_logits[no_target_bin & (bins == time_bin)]
        if values.numel():
            budget_terms.append(functional.softplus(torch.topk(values, min(4, values.numel())).values).mean())
    false_budget = torch.stack(budget_terms).mean() if budget_terms else node_logits.new_zeros(())
    return node_loss + coverage_loss + 0.25 * edge_loss + 0.50 * false_budget
