import torch
import torch.autograd as autograd
import torch.optim as optim

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from fairseq.models.roberta import RobertaModel

torch.manual_seed(1)

import torch
import torch.nn as nn
roberta = RobertaModel.from_pretrained('/Users/apple/PycharmProjects/fairseq/fairseq/models/roberta/roberta.base', checkpoint_file='model.pt')
roberta.eval()

assert isinstance(roberta.model, torch.nn.Module)



# how to use roBERTa
# bpe extract word byte pair encoding
#

"""
Sort according to length
"""
def sort_batch(data, label, length):
  batch_size = data.size(0)

  # first chaneg data intp numpy and sort to get index
  inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
  data = data[inx]
  label = label[inx]
  length = length[inx]

  # length to transfer to list not using torch.Tensor
  length = list(length.numpy())
  return (data, label, length)


class BiLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout=0.5):

    super(BiLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.num_layers = num_layers
    if (biFlag):
      self.bi_num = 2
    else:
      self.bi_num = 1
    self.biFlag = biFlag
    # change device
    self.device = torch.device("cuda")

    # define LSTM input output layer_number  batch_first and dropout portion
    self.layer1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, \
                          num_layers=num_layers, batch_first=True, \
                          dropout=dropout, bidirectional=biFlag)
    # define output layer and use log_softmax to output
    self.layer2 = nn.Sequential(
      nn.Linear(hidden_dim * self.bi_num, output_dim),
      nn.LogSoftmax(dim=2)
    )

    self.to(self.device)

  def init_hidden(self, batch_size):
    # define the hidden state
    return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device))

  def forward(self, x, y, length):
    # input original x, label y and length
    batch_size = x.size(0)
    max_length = torch.max(length)
    # truncked according to the maxium length
    x = x[:, 0:max_length, :]
    y = y[:, 0:max_length]
    x, y, length = sort_batch(x, y, length)
    x, y = x.to(self.device), y.to(self.device)
    # pack sequence
    x = pack_padded_sequence(x, length, batch_first=True)

    # run the network
    hidden1 = self.init_hidden(batch_size)
    out, hidden1 = self.layer1(x, hidden1)
    # out,_=self.layerLSTM(x) is also ok if you don't want to refer to hidden state
    # unpack sequence
    out, length = pad_packed_sequence(out, batch_first=True)
    out = self.layer2(out)

    # return the node and prediction output and length
    return y, out, length


# roberta = RobertaModel.from_pretrained('/path/to/roberta.base', checkpoint_file='model.pt')
#
#
#
# all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
# assert len(all_layers) == 25
# assert torch.all(all_layers[-1] == last_layer_features)
# # roberta.eval()  # disable dropout (or leave in train mode to finetune)


