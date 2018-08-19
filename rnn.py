import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt

np.random.seed(69)
torch.manual_seed(69)

EPOCH=300
HIDDEN_DIM=32
NUM_LAYERS=128
BATCH_SIZE=72

class MultiRNN(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1):
		
		super(MultiRNN, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.hidden = self.init_hidden()
		self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
		self.linear = nn.Linear(hidden_dim, input_dim)

	
	def init_hidden(self): 
		return (torch.zeros(self.num_layers,1,self.hidden_dim), torch.zeros(self.num_layers,1,self.hidden_dim))
 
	def forward(self, data):
		rnn_out, hidden = self.rnn(data)
		vecs = self.linear(rnn_out)
		return vecs