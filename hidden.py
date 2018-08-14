import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
from gru import MultiGRU
from lstm import MultiLstm

np.random.seed(69)
torch.manual_seed(69)

def tt(modelt, hidden, lyr, epochh):
	out=0
	for i in range(10):
		if modelt=='gru': model = MultiGRU(8,hidden,lyr,1)
		elif modelt=='lstm': model = MultiLstm(8,hidden,lyr,1)
		print("   Run:", i)
		loss_function = nn.L1Loss(size_average=True)
		optimizer = optim.Adam(model.parameters())
		model.zero_grad()

		with torch.no_grad():
			data = torch.load('train_X.pt')
			labels = torch.load('train_y.pt')
			val_x = torch.load('val_X.pt')
			val_y = torch.load('val_y.pt')
		
		for epoch in range(epochh):
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
			with torch.no_grad():
				out_val = model(val_x)
				loss_val = loss_function(out_val, val_y)
				valloss += float(loss_val.data)
			print('      Epoch:', epoch+1,'/', epochh, '| Training Loss:', cumloss, '| Validation Loss:', valloss)
		out+=valloss
	return out/10


nums = [1,16,32,64,128,256]
avgs_gru = []
avgs_lstm = []
for dim in nums:
	print("GRU dims:", dim)
	if dim == 64: epoch = 100
	elif dim > 64: epoch = 200
	else: epoch = 60 
	avgs_gru.append(tt('gru', dim, 32, epoch))
for dim in nums:
	print("LSTM dims:", dim)
	if dim == 64: epoch = 100
	elif dim > 64: epoch = 200
	else: epoch = 60 
	avgs_lstm.append(tt('lstm', dim, 32, epoch))
print(avgs_gru)
print(avgs_lstm)


