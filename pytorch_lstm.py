import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

EPOCH=50
HIDDEN_DIM=8
NUM_LAYERS=5

torch.manual_seed(69)

class MultiLstm(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size):
		super(MultiLstm, self).__init__()	
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers)
		self.hidden = self.init_hidden() 	#(num_layers, minibatch_size, hidden_dim)
		self.linear = nn.Linear(batch_size*hidden_dim, 1)
	
	def init_hidden(self): return (torch.zeros(self.num_layers,1,self.hidden_dim), torch.zeros(self.num_layers,1,self.hidden_dim))
 
	def forward(self, data):
		lstm_out, self.hidden = self.lstm(data, self.hidden)
		print(lstm_out.shape)
		out = self.linear(lstm_out.view(-1)) 
		return out

model = MultiLstm(8,HIDDEN_DIM,10)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

with torch.no_grad():
	train_X	= torch.from_numpy(np.load('train_X.npy'))
	train_y	= torch.from_numpy(np.load('train_y.npy'))
	test_X	= torch.from_numpy(np.load('test_X.npy'))
	test_y	= torch.from_numpy(np.load('test_y.npy'))

	data = train_X
	labels = train_y

	print(model(data[:5]).view(1,1,-1))

for epoch in range(EPOCH):
	cumloss = 0
	for i in range(data.shape[0]):
		model.zero_grad()
		model.hidden = model.init_hidden()

		out = model(data[i].view(1,1, data.shape[2]))

		loss = loss_function(out.view(1), labels[i])
		loss.backward()
		optimizer.step()
		cumloss += float(loss.data)

	print('Epoch:', epoch+1,'/', EPOCH, '| Epoch Loss:', cumloss)

with torch.no_grad(): print(model(data[:5]).view(1,1,-1)); print(labels[:5])





