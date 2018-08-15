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

def tt(modelt, data, labels):
	out=0
	for i in range(10):
		if modelt=='gru': model = MultiGRU(8,32,5,1)
		elif modelt=='lstm': model = MultiLstm(8,32,5,1)
		print("   Run:", i)
		loss_function = nn.L1Loss(size_average=True)
		optimizer = optim.Adam(model.parameters())
		model.zero_grad()

		with torch.no_grad():
			val_x = torch.load('valx.pt')
			val_y = torch.load('valy.pt')
		
		for epoch in range(60):
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
			print('      Epoch:', epoch+1,'/', 60, '| Training Loss:', cumloss, '| Validation Loss:', valloss)
		out+=valloss
	return out/10

train1 = torch.load('train1.pt')
train2 = torch.load('train2.pt')
train3 = torch.load('train3.pt')
train4 = torch.load('train4.pt')

trainy1 = torch.load('trainy1.pt')
trainy2 = torch.load('trainy2.pt')
trainy3 = torch.load('trainy3.pt')
trainy4 = torch.load('trainy4.pt')

sets = [(train1, trainy1), (train2, trainy2), (train3, trainy3), (train4, trainy4)]
avgs_gru = []
avgs_lstm = []
for data,lables in sets:
	print("GRU Layers:", lyr) 
	avgs_gru.append(tt('gru', data, labels))
for lyr in nums:
	print("LSTM Layers:", lyr)
	avgs_lstm.append(tt('lstm', data, labels))
print(avgs_gru)
print(avgs_lstm)
