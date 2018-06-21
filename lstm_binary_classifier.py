import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

EPOCH=50
HIDDEN_DIM=32
NUM_LAYERS=5
BATCH_SIZE=72

torch.manual_seed(69)

class MultiLstm(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1):
		super(MultiLstm, self).__init__()

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.hidden =self.init_hidden()

		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
		self.linear = nn.Linear(hidden_dim, 1)

	
	def init_hidden(self): 
		return (torch.zeros(self.num_layers,1,self.hidden_dim), torch.zeros(self.num_layers,1,self.hidden_dim))
 
	def forward(self, data):
		lstm_out, hidden = self.lstm(data)
		vecs = self.linear(lstm_out)
		out = F.tanh(vecs)
		return vecs
	
model = MultiLstm(90000, HIDDEN_DIM, 5, 1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

with torch.no_grad():

	train_X	= torch.from_numpy(np.load('train_X.npy'))
	train_y	= torch.from_numpy(np.load('train_y.npy'))
	test_X	= torch.from_numpy(np.load('test_X.npy'))
	test_y	= torch.from_numpy(np.load('test_y.npy'))
	
	unbatched_data = train_X
	unbatched_labels = train_y

	temp_d, temp_l = train_X.split(train_X.shape[0]//72, dim=0), train_y.split(train_y.shape[0]//72, dim=0)
	data, labels = torch.cat(temp_d[:-1], dim=1), torch.cat(temp_l[:-1])
	print(data.shape, labels.shape)

	print(model(data[:5]).view(1,1,-1))

for epoch in range(EPOCH):
	cumloss = 0
	for i in range(data.shape[1]):
		model.zero_grad()
		model.hidden = model.init_hidden()

		out = model(data[:,i,:].view(-1,1, data.shape[2]))

		loss = loss_function(out.view(1), labels[i])
		loss.backward()
		optimizer.step()
		cumloss += float(loss.data)

	print('Epoch:', epoch+1,'/', EPOCH, '| Epoch Loss:', cumloss)

with torch.no_grad(): print(model(data[:5]).view(1,1,-1)); print(labels[:5])