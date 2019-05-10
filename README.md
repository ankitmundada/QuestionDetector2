# QuestionDetection using Deep Learning

## Model architecture:

This is a LSTM based Recurrent Network. The many-to-one approach is used to predict the last label of an input sentence, the label being either `question` or `not-a-`question`.

1. Instead of using simple one-hot-encodings for the vocabulary, it uses word embeddings to represent high-dimentional patterns.
2. The network uses `packed padded sequences` because of variability in input lengths. The packed padded inputs help in speeding up the training since lot of processing for `zero-padding` is not done. 
3. This is binary classification architecture, in which an input sequence is classified as either a `question` or `not-a-question`
4. The final probabilties are calculated using `softmax`
5. All the `params` can be found in `configs.json` file
```python
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(QuestionDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 3)
        self.fc1 = nn.Linear(hidden_dim, 1) 
```

## Training:

Run the following command to train the model. Note: All the `params` are loaded from `config.json`
```python
python train.py --config config.json
```

## Dataset:

### Pre-processing:
The input data is expected to be in the following format:

1. Each line in the `data_file` is a `sample_input`
2. The last word of each line is either a `|` (`not-a-question`) or `?` (`question`) 
3. The raw text data can be cleaned into a usable format using the script in `conversion/text_to_data.py` which uses **multiprocessing** to fasten the process


## Tensorboard Logging:

All the `logs` are saved in the `saved` folder. Project uses `tensorboard` to write the logs and so `tensorboardX` needs to be installed in your environment

