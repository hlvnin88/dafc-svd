"""BiLSTM encoder cho F2."""
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128, out_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, out_dim), nn.ReLU(),
                                   nn.Dropout(dropout))
    def forward(self, x, lengths):
        e = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h_cat)