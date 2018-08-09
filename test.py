import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from gru import MultiGRU
from lstm import MultiLstm

np.random.seed(69)
torch.manual_seed(69)

EPOCH=100
HIDDEN_DIM=32
NUM_LAYERS=32
BATCH_SIZE=72

def layerTest():
for lyrs in [1,8,16,32,64,128]:
	for _ in range(10):
		
gru = MultiGRU(8, HIDDEN_DIM, NUM_LAYERS, 1)
lstm = MultiLstm()
loss_function = nn.L1Loss(size_average=True)
optimizer = optim.Adam(model.parameters())
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

with torch.no_grad():
	data = torch.load('train_X.pt')
	labels = torch.load('train_y.pt')
	val_x = torch.load('val_X.pt')
	val_y = torch.load('val_y.pt')
	print(model(data[:3,0,:].view(3,1,data.shape[2])))
	lossx = [i for i in range(EPOCH)]
	valx= [i for i in range(EPOCH)]
	lossy,valy= [],[]

for epoch in range(EPOCH):
	cumloss=0
	valloss=0
	model.zero_grad()
	optimizer.zero_grad()
	model.hidden = model.init_hidden()

	out = model(data)
	loss=loss_function(out, labels)
	loss.backward()
	optimizer.step()
	cumloss += float(loss.data)
	lossy.append(cumloss)
	with torch.no_grad():
		out_val = model(val_x)
		loss_val = loss_function(out_val, val_y)
		valloss += float(loss_val.data)
		valy.append(valloss)
	print('Epoch:', epoch+1,'/', EPOCH, '| Training Loss:', cumloss, '| Validation Loss:', valloss)



	# for i in range(data.shape[1]):
	# 	model.zero_grad()
	# 	optimizer.zero_grad()

	# 	model.hidden = model.init_hidden()

	# 	out = model(data[:,i,:].view(data.shape[0],1,data.shape[2]))

	# 	loss = loss_function(out, labels[:,i,:].view(labels.shape[0],1,labels.shape[2]))
	# 	loss.backward()
	# 	optimizer.step()
	# 	cumloss += float(loss.data)

	# print('Epoch:', epoch+1,'/', EPOCH, '| Epoch Loss:', cumloss)

with torch.no_grad(): 
	print(model(val_x[:3,0,:].view(3,1,data.shape[2])))
	print(val_y[:3,0,:].view(3,1,labels.shape[2]))
	torch.save(model.state_dict(), 'gru.pt')

	plt.plot(valx, valy, 'b')
	plt.plot(lossx, lossy, 'r')
	plt.show()
