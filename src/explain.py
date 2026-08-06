"""Interpretability: Integrated Gradients over channels and time steps.

Using captum's IntegratedGradients gives you a direct, apples-to-apples
comparison point against the paper, since they benchmark MaGNet's own
interpretation model against IntGradients (Section 6 of the paper).
"""
import torch
from captum.attr import IntegratedGradients


def channel_time_attributions(model, x: torch.Tensor, target_head: str, target_class: int):
    """Compute per-channel, per-timestep attribution scores.

    Args:
        model: trained MultiTaskRNN
        x: input batch, shape (batch, seq_len, n_channels)
        target_head: "inseq" or "odor", picks which output head to explain
        target_class: which class index within that head to explain

    Returns:
        attributions: same shape as x, (batch, seq_len, n_channels)
    """
    def forward_fn(inputs):
        inseq_logits, odor_logits = model(inputs)
        return inseq_logits if target_head == "inseq" else odor_logits

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(x, target=target_class)
    return attributions


def channel_importance_summary(attributions: torch.Tensor):
    """Collapse (batch, seq_len, n_channels) attributions down to a
    per-channel importance score, averaged over batch and time.
    Useful for comparing against MaGNet's node-importance rankings.
    """
    return attributions.abs().mean(dim=(0, 1))
