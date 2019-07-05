import torch.nn as nn


class RNNDropout(nn.Dropout):
    """Dropout for RNN."""

    def forward(self, sequences_batch):
        """Masking whole hidden vector for tokens."""
        # B: batch size
        # L: sequence length
        # D: hidden size

        # sequence_batch: BxLxD
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch
