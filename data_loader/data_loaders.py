import os
import torch
import pickle
import numpy as np
from base import BaseDataLoader
from dataset import datasets
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack


class QuestionsLoader(BaseDataLoader):
    """
    Dataloader for words with single label output of whether the sentence is a question or not
    """
    def __init__(self, data_dir, mode, batch_size=128, shuffle=True, num_workers=4):

        self.data_dir = data_dir

        if mode=="train":
            self.text_data_file = os.path.join(data_dir, 'train.txt')
        elif mode=="val":
            self.text_data_file = os.path.join(data_dir, 'val.txt')
        elif mode=="test":
            self.text_data_file = os.path.join(data_dir, 'test.txt')
        else:
            raise ValueError("Incorrect mode provided")

        self.dataset = datasets.QuestionsDataset(self.text_data_file, word_to_idx=self._get_dict())

        super(QuestionsLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers, collate_fn=collate_fn)

    def _get_dict(self):
        dict_path = os.path.join(self.data_dir, 'word_to_idx.pickle')

        if os.path.exists(dict_path):
            print("Loading pre-written 'word_to_idx' dictionary")
            with open(dict_path, 'rb') as inp:
                word_to_idx = pickle.load(inp)
                self.vocab_size = len(word_to_idx)
                return word_to_idx

        print("Pre-written 'word_to_idx' not found. Creating one")

        vocab = {'UNK'}
        with open(self.text_data_file, 'r') as inp:
            line = inp.readline()
            while line:
                line = line.lower().strip()
                vocab.update(line.split())
                line = inp.readline()

        word_to_idx = {w: idx+1 for idx, w in enumerate(vocab)}
        self.vocab_size = len(word_to_idx)

        print("Saving 'word_to_idx' dictionary")
        with open(dict_path, 'wb') as out:
            pickle.dump(word_to_idx, out, protocol=pickle.HIGHEST_PROTOCOL)
        return word_to_idx

    def get_vocab_size(self):
        return self.vocab_size


def collate_fn(batch):
    """
    Custom collate_fn is required since samples in a batch are not of same length,
    hence they need to be padded
    """
    sents   = [b["words"] for b in batch]
    labels  = np.asarray([b["label"] for b in batch], dtype=np.float32)

    sent_lengths = np.asarray([words.shape[0] for words in sents], dtype=np.int32)
    sort_idx = np.argsort(sent_lengths)[::-1]
    sent_lengths = sent_lengths[sort_idx] #sorted
    sents_sorted = [sents[i] for i in sort_idx.tolist()]
    labels = labels[sort_idx]
    max_length = max(sent_lengths)
    for i in range(len(sents)):
        sents[i] = np.append(sents[i], np.zeros(max_length - sents[i].shape[0]))

    packed_batch = pack(torch.tensor(sents), torch.tensor(sent_lengths), batch_first=True)
    
    return packed_batch, torch.tensor(labels)
