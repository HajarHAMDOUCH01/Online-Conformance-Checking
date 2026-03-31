import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMFitnessModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, input_size=185, hidden_size=64, output_size=1, num_layers=2):
        super(LSTMFitnessModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths, h=None, c=None):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        if h is not None and c is not None:
            packed_out, (last_h,last_c) = self.lstm(packed, (h,c))
        else:
            packed_out, (last_h,last_c) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1,1, out.size(2))
        last_out = out.gather(1, idx).squeeze(1)
        return self.sigmoid(self.fc(last_out)), (last_h,last_c)

