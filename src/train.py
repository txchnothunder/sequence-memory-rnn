"""Training entry point.

Run with: python -m src.train --config configs/baseline.yaml

The loop itself is ready to go; what's missing is the dataset/loader,
which depends on preprocessing.py being filled in after the data
audit.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from src.models import MultiTaskRNN, multitask_loss
from src.utils import set_seed, load_config, get_device


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()

    # TODO: replace with a real Dataset built from src/preprocessing.py
    # once trial segmentation and label extraction are implemented.
    raise NotImplementedError(
        "Wire up a Dataset/DataLoader here once preprocessing.py is done. "
        "Model, loss, and training loop below are ready to use as-is."
    )

    model = MultiTaskRNN(
        input_size=None,  # TODO: n_channels from your data
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        rnn_type=cfg["model"]["type"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        # TODO: loop over train_loader
        # for x, inseq_labels, odor_labels in train_loader:
        #     x = x.to(device)
        #     inseq_labels = inseq_labels.to(device)
        #     odor_labels = odor_labels.to(device)
        #     optimizer.zero_grad()
        #     inseq_logits, odor_logits = model(x)
        #     loss, loss_inseq, loss_odor = multitask_loss(
        #         inseq_logits, odor_logits, inseq_labels, odor_labels,
        #         cfg["train"]["loss_weight_inseq"], cfg["train"]["loss_weight_odor"],
        #     )
        #     loss.backward()
        #     optimizer.step()

        # TODO: validation pass + early stopping using
        # cfg["train"]["early_stopping_patience"]
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    train(args.config)
