import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class MultiLstm(nn.Module):

	def __init__(self, input_dim, hidden_dim):
		super(MultiLstm, self).__init__()
		self.hidden_dim=hidden_dim
		self.lstm=nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
		self.hidden=(torch.zeros(1,1,self.hidden_dim), 	#(num_layers, minibatch_size, hidden_dim)
					 torch.zeros(1,1,self.hidden_dim))
 
	def forward(self, data):
		lstm_out, self.hidden=self.lstm(data, self.hidden)
		return lstm_out

model=MultiLstm(8,8)
loss_function=nn.MSELoss()
optimizer=optim.Adam(model.parameters())

with torch.no_grad():
	train_X=torch.from_numpy(np.load('train_X.npy'))
	train_y=torch.from_numpy(np.load('train_y.npy'))
	test_X=torch.from_numpy(np.load('test_X.npy'))
	test_y=torch.from_numpy(np.load('test_y.npy'))

	print(model(train_X[:5]))

for epoch in range(50):
	for










		

		
