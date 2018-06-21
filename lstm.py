import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

EPOCH=50
HIDDEN_DIM=64
NUM_LAYERS=50
BATCH_SIZE=72

torch.manual_seed(69)

class MultiLstm(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers=1, batch_size=1):
		
		super(MultiLstm, self).__init__()

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.hidden = self.init_hidden()

		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
		self.linear = nn.Linear(hidden_dim, 8)

	
	def init_hidden(self): 
		return (torch.zeros(self.num_layers,1,self.hidden_dim), torch.zeros(self.num_layers,1,self.hidden_dim))
 
	def forward(self, data):
		lstm_out, hidden = self.lstm(data)
		vecs = self.linear(lstm_out)
		return vecs
	
model = MultiLstm(8, HIDDEN_DIM, NUM_LAYERS, 1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

with torch.no_grad():
	data = torch.load('train_X.pt')
	labels = torch.load('train_y.pt')

	data = torch.load('test_X.pt')
	labels = torch.load('test_y.pt')

	print(model(data[:3,0,:].view(3,1,data.shape[2])))

for epoch in range(EPOCH):
	cumloss = 0
	for i in range(data.shape[1]):
		model.zero_grad()
		model.hidden = model.init_hidden()

		out = model(data[:,i,:].view(data.shape[0],1,data.shape[2]))

		loss = loss_function(out, labels[:,i,:].view(labels.shape[0],1,labels.shape[2]))
		loss.backward()
		optimizer.step()
		cumloss += float(loss.data)

	print('Epoch:', epoch+1,'/', EPOCH, '| Epoch Loss:', cumloss)

with torch.no_grad(): 
	print(model(data[:3,0,:].view(3,1,data.shape[2])))
	print(model(labels[:3,0,:].view(3,1,labels.shape[2])))
	torch.save(model.state_dict(), 'lstm.pt')