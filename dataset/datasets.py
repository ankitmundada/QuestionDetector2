import numpy as np
from torch.utils.data import Dataset


class QuestionsDataset(Dataset):
    """
    Sentence Dataset
    """

    def __init__(self, text_data, word_to_idx=None):
        """
        """
        with open(text_data, 'r') as inp:
            self.sents = inp.readlines()
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = self.sents[idx].split()
        label = 1 if words[-1] == '?' else 0

        words = [self._get_word_idx(w) for w in words[:-1]]
        item = {
                "words": np.asarray(words, dtype=np.float32),
                "label": label
                }
        return item

    def _get_word_idx(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else: 
            return self.word_to_idx['UNK']

