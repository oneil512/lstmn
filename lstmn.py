from __future__ import division
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import scipy.misc as smp
from pytorch_pretrained_bert.tokenization import BertTokenizer

###
# Author: Clay O'Neil
# Implementation of Long Short-Term Memory-Networks for Machine Reading by Jianpeng Cheng, Li Dong and Mirella Lapata
# https://arxiv.org/pdf/1601.06733.pdf
###

torch.manual_seed(11)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class LSTMN(nn.Module):

  ### PARAMETERS 
  # dim is number of hidden units
  # max_len is the maximum length of the input
  # batch_size is length of minibatch
  # output_size is dim of final output
  # init_hidden (optional) is the initial hidden state
  # init_cell (optional) is the initial cell state
  ### RETURNS
  # outputs, size (batch_size, output_size)
  # hidden_states, size (batch_size, max_length, dim)
  ###

  def __init__(self, dim, max_len, batch_size, output_size, init_hidden=0, init_cell=0):
    super(LSTMN, self).__init__()

    self.max_len = max_len
    self.dim = dim
    self.output_size = output_size
    self.batch_size = batch_size

    # Weight matrix for each of the 4 gates
    self.W = torch.randn(4 * dim, max_len + dim, device=device, dtype=torch.float)

    # Bias matrix
    self.b = torch.randn(4 * dim, device=device, dtype=torch.float)

    # Set forget bias to 1 per http://proceedings.mlr.press/v37/jozefowicz15.pdf
    self.b[:dim] = 1

    # initial hidden state
    self.h = torch.randn(dim, device=device, dtype=torch.float) if init_hidden == 0 else init_hidden

    # initial cell state
    self.c = torch.randn(dim, device=device, dtype=torch.float) if init_cell == 0 else init_cell

    # Attention weights
    self.W_h = torch.randn(10, dim, device=device, dtype=torch.float)
    self.W_x = torch.randn(10, max_len, device=device, dtype=torch.float)
    self.W_ht = torch.randn(10, dim, device=device, dtype=torch.float)

    self.v = torch.randn(10, device=device, dtype=torch.float)

    self.ht = torch.randn(dim, device=device, dtype=torch.float)
    self.ct = torch.randn(dim, device=device, dtype=torch.float)

    # Projection layer
    self.projection = torch.nn.Linear(dim, output_size)

  # x is batch of input column vectors shape (batch_size, max_len)
  def forward(self, x):
    batch_hidden_states = []
    outputs = []

    for i in range(self.batch_size):
      hidden_states = []
      past_ht = []
      past_x = []
      past_c = []

      x_i = x[i]

      for x_index, x_t in enumerate(x_i):
        attention_vector = []
        past_x.append(x_t)
        

        # Iterate through past hidden states and calculate attention vector
        for k, h in enumerate(hidden_states):
          a_t = self.v @ (nn.Tanh() (self.W_h @ h + self.W_x @ past_x[k] + self.W_ht @ past_ht[k]))
          attention_vector.append(a_t)

        attention_vector = torch.Tensor(attention_vector)
        attention_softmax = torch.nn.Softmax() (attention_vector)

        if len(hidden_states) > 0:
          ht = 0
          ct = 0
          for k, s in enumerate(attention_softmax):
            ht += s * hidden_states[k]
            ct += s * past_c[k]
            
          self.ht = torch.Tensor(ht)
          self.ct = torch.Tensor(ct)

          att = smp.toimage(attention_softmax.detach().view(-1,1))
          smp.imsave('plots/att-' + str(i) + str(x_index) + '.png' ,att)

        concat_input = torch.cat((self.ht, x_t), 0).view(-1, 1)
        whx = self.W.mm(concat_input).view(-1) + self.b

        f_t = nn.Sigmoid() (whx[:self.dim])
        o_t = nn.Sigmoid() (whx[self.dim:self.dim * 2])
        i_t = nn.Sigmoid() (whx[self.dim * 2:self.dim * 3])
        ch_t = nn.Tanh() (whx[self.dim * 3:])

        self.c = f_t * self.ct + i_t * ch_t
        self.h = o_t * (nn.Tanh() (self.c))

        past_ht.append(self.ht)
        hidden_states.append(self.h)
        past_c.append(self.c)
      

      logits = self.projection(self.h)
      output = torch.nn.Softmax()(logits)
      outputs.append(output)


      batch_hidden_states.append(hidden_states)

    return outputs, batch_hidden_states
      
def one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

# Cross Entropy Log Loss
def loss_fn(pred, gt):
  r = 0
  for i, row in enumerate(pred):
    r += torch.log(row.dot(gt[i].view(-1)) + 1e-8)
  return -1 * (r / len(pred))

max_len = 10
dim = 64
batch_size = 16
output_size = 10
      
lstmn = LSTMN(dim, max_len, batch_size, output_size)

# Create dataset
x_data = []
y_data = []
for i in range(200):
  x = random.choices(range(10), k=max_len)
  y = [1 if x[4] > 5 else 6]
  x_data.append(one_hot(x, 10))
  y_data.append(one_hot(y, 10))

dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.Adam(lstmn.parameters(), lr=3e-4)
epochs = 50
losses = []

for epoch in range(epochs):
  for batch_idx, batch in enumerate(dataloader):
    x = batch[0]
    y = batch[1]
    outputs, hidden_states = lstmn(x)
    loss = loss_fn(outputs, y)
    print(batch_idx + epoch * len(dataset), loss.item())
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

t = np.linspace(0, 1000, len(losses))
y = np.cos(np.pi * (t / len(losses)))

plt.scatter(t, losses, c=y, s=1)

plt.axis([-1, 1000, -.01, 3])
plt.xlabel('batches', fontsize=14, color='red')
plt.ylabel('loss', fontsize=14, color='red')
plt.show()
