import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class QuestionDetector(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(QuestionDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 3)
        self.fc1 = nn.Linear(hidden_dim, 1) 

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence[0])
        packed_embeds = torch.nn.utils.rnn.PackedSequence(embeds, sentence[1])
        out, _ = self.lstm(packed_embeds)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out    = self.fc1(out[:, -1, :])
        out    = F.sigmoid(out)
        return out

    def simple_elementwise_apply(fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

