import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class QuestionDetector(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(QuestionDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1) 

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        out    = self.fc1(out.view(len(sentence), -1))
        out    = F.log_softmax(out, dim=1)
        return out

    def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)
