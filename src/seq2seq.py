# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2018-07-24 19时19分49秒
# Last modified : 2018-07-24 19时23分07秒
#     File Name : seq2seq.py
#          Desc :

import torch

# model/Encoder.py
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence

class VanillaEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(VanillaEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, output_size)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        packed_outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(packed_outputs)
        return outputs, hidden

if __name__ == "__main__":
    pass
