

import torch.nn as nn
import torch
import re
from utils import util


class EntModel(nn.Module):
    """
    Entity Model: RNN
    """
    def __init__(self,
                 word_char_embedding,
                 word_char_emb_size,
                 hidden_size,
                 parse_lstm_size,
                 tag_emb_size,
                 tag_size,
                 chunk_vocab,
                 num_layers=1,
                 use_cuda=False,
                 bidirectional=True,
                 dropout=0.5):
        super(EntModel, self).__init__()
        self.word_char_embedding = word_char_embedding
        self.word_char_emb_size = word_char_emb_size
        self.use_cuda = use_cuda
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.parse_lstm_size = parse_lstm_size
        self.tag_size = tag_size
        self.chunk_vocab = chunk_vocab
        self.dropout = nn.Dropout(dropout)
        self.tag_emb_size = tag_emb_size

        self.chunk_num, self.idx2chunk = self.parse_chunk_vocab(chunk_vocab) # idx2chunk is list type where every element is entity type, such as PERSON
        self.chunk2id = {v: k for k, v in enumerate(self.idx2chunk)}

        self.tag_embeddings = nn.Embedding(tag_size, tag_emb_size)

        self.encoder = nn.LSTM(word_char_emb_size + self.parse_lstm_size,
                               self.hidden_size // 2,
                               num_layers=1,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)


    def parse_chunk_vocab(self, chunk_vocab):
        entity_set = set()
        for tag in chunk_vocab.id2item:
            if tag == 'O':
                continue
            _, tag_type = util.parse_tag(tag)
            entity_set.add(tag_type)
        entity_id2item = list(entity_set)
        return len(entity_id2item), entity_id2item


