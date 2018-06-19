import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#ADJUST THIS
EPOCH = 10
HIDDEN_DIM = 8 
NUM_LAYERS = 5
INPUT_SIZE = 8 #you have to change 2d image to 1d

torch.manual_seed(69)

class ClassLSTM(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers):
		super(ClassLSTM, self).__init__()	
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers)
		self.hidden = self.init_hidden() 	#(num_layers, minibatch_size, hidden_dim)
		self.linear = nn.Linear(input_dim, 1)
	
	def init_hidden(self): return (torch.zeros(self.num_layers,1,self.hidden_dim), torch.zeros(self.num_layers,1,self.hidden_dim))
 
	def forward(self, data):
		lstm_out, self.hidden = self.lstm(data, self.hidden)
		temp = self.linear(lstm_out)
		out = F.tanh(temp)
		return out

model = ClassLSTM(INPUT_SIZE, HIDDEN_DIM, NUM_LAYERS) #ADJUST THIS for input dim
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

with torch.no_grad():
	# import data, store data as train_X, labels as train_y
	# train_X must have dimensions (num pics, MINIBATCH NUM, 90000). If minibatch means nothing to you, set MINIBATCH NUM=1
	# train_y should be (num pics, label) where label either {-1,1}

	# here's some example data:
	train_X	= torch.from_numpy(np.load('train_X.npy'))
	train_y = torch.from_numpy(np.random.randint(-1,2,size=train_X.shape[0])).type(torch.FloatTensor)
	# train_y	= torch.from_numpy(np.load('train_y.npy'))
	# test_X	= torch.from_numpy(np.load('test_X.npy'))
	# test_y	= torch.from_numpy(np.load('test_y.npy'))	
	
	print(model(train_X[:5])) #prints prediction of first 5 measurements with untrained net for comparison later
	
for epoch in range(EPOCH):
	print('Epoch:', epoch+1,'/', EPOCH)
	for i in range(train_X.shape[0]):
		model.zero_grad()
		model.hidden = model.init_hidden()
		out = model(train_X[i].view(1,1, train_X.shape[2]))
		loss = loss_function(out.view(1), train_y[i])
		loss.backward()
		optimizer.step()

with torch. no_grad(): print(model(train_X[:5])) #prints prediction of trained net