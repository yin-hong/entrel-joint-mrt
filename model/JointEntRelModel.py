

import torch
import torch.nn as nn
from module.EntModel import EntModel
from module.RelModel import RelModel


class JointEntRelModel(nn.Module):
    """
    End-to-end entities and relations extraction model. This model jointly extract entities and relation
    through shared parameters. Entity model employs RNN while Relation model uses CNN .
    """
    def __init__(self,
                 word_char_embedding,
                 word_char_emb_size,
                 out_channels,
                 kernel_size,
                 hidden_size,
                 parse_lstm_size,
                 tag_emb_size,
                 position_emb_size,
                 tag_size,
                 max_sent_len,
                 chunk_vocab,
                 N_ID,
                 num_layers=1,
                 use_cuda=False,
                 bidirectional=True,
                 win=15,
                 sch_k=0.5,
                 dropout=0.5):
        super(JointEntRelModel, self).__init__()
        self.use_cuda = use_cuda
        self.sch_k = sch_k

        self.entity_model = EntModel()
        self.rel_model = RelModel()

    def forward(self, X, X_char, X_lstm_h, X_len, Y, Y_rel, i_epoch=0):

        pass


    def forward_sample(self, X, X_char, X_lstm_h, X_mask, sample_Y, Y_rel):
        pass

