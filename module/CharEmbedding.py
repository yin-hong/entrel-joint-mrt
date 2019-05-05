
import torch.nn as nn
import torch.nn.functional as F
import torch


class CharEmbedding(nn.Module):
    """
    Generate char embedding with convolutional neural network
    """
    def __init__(self, vocab_size, embedding_size,
                 out_channels, kernel_sizes, padding_idx=0, dropout=0.5):
        super(CharEmbedding, self).__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, out_channels, (K, embedding_size), padding=(K-1, 0))
                                     for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # these characters is made up of word.
        # remember: we don't want to obtain one vector per character. Actually, we just want obtain one vector
        # with all character, because we will concatenate this vector with word vector.
        x = self.char_embeddings(X)  # X shape: (num_of_character, dimension)
        x = self.dropout(x)
        x = x.unsqueeze(1)  # shape: (N, C, W, D) where C=1
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # shape: (N, C1, W1)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # shape: (N, C1)
        x = torch.cat(x, 1)
        return self.dropout(x)



