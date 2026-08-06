"""Multi-task RNN: shared recurrent trunk, two heads.

Head 1: InSeq/OutSeq (binary)
Head 2: Odor identity A-E (5-class)

This part doesn't depend on the data audit results, input_size just
needs to match your number of channels/features once known.
"""
import torch
import torch.nn as nn


class MultiTaskRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        rnn_type: str = "gru",
        bidirectional: bool = False,
        dropout: float = 0.2,
        n_odor_classes: int = 5,
    ):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        rnn_out_size = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.inseq_head = nn.Linear(rnn_out_size, 2)
        self.odor_head = nn.Linear(rnn_out_size, n_odor_classes)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, n_channels)"""
        out, hidden = self.rnn(x)
        # Use the last time step's hidden representation.
        # Attention pooling is a natural upgrade later, both for
        # accuracy and for a built-in interpretability signal.
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)

        inseq_logits = self.inseq_head(last_out)
        odor_logits = self.odor_head(last_out)
        return inseq_logits, odor_logits


def multitask_loss(inseq_logits, odor_logits, inseq_labels, odor_labels, w_inseq=1.0, w_odor=1.0):
    ce = nn.CrossEntropyLoss()
    loss_inseq = ce(inseq_logits, inseq_labels)
    loss_odor = ce(odor_logits, odor_labels)
    return w_inseq * loss_inseq + w_odor * loss_odor, loss_inseq, loss_odor
