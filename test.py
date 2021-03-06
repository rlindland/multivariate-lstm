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

# EPOCH=60
# HIDDEN_DIM=32
# NUM_LAYERS=32
# BATCH_SIZE=72

# def layerTest():
# 	avgs=[]
# 	nums=[1,8,16,32,64,128]
# 	for lyrs in [1,8,16,32,64,128]:
# 		print('Num Layers:', lyrs)
# 		avg=0
# 		for r in range(10):
# 			print('   Run:', r)
# 			model = MultiGRU(8, 32, lyrs, 1)
# 			loss_function = nn.L1Loss(size_average=True)
# 			optimizer = optim.Adam(model.parameters())
# 			model.zero_grad()

# 			with torch.no_grad():
# 				data = torch.load('train_X.pt')
# 				labels = torch.load('train_y.pt')
# 				val_x = torch.load('val_X.pt')
# 				val_y = torch.load('val_y.pt')

			
# 			for epoch in range(60):
# 				cumloss=0
# 				valloss=0
# 				model.zero_grad()
# 				optimizer.zero_grad()
# 				model.hidden = model.init_hidden()

# 				out = model(data)
# 				loss=loss_function(out, labels)
# 				loss.backward()
# 				optimizer.step()
# 				cumloss += float(loss.data)
# 				with torch.no_grad():
# 					out_val = model(val_x)
# 					loss_val = loss_function(out_val, val_y)
# 					valloss += float(loss_val.data)
# 				print('      Epoch:', epoch+1,'/', EPOCH, '| Training Loss:', cumloss, '| Validation Loss:', valloss)
# 			avg+=valloss
# 		avgs.append(avg/10)

# 	avgs2=[]
# 	nums2=[1,8,16,32,64,128]
# 	for lyrs in [1,8,16,32,64,128]:
# 		print('Num Layers:', lyrs)
# 		avg=0
# 		for r in range(10):
# 			print('   Run:', r)
# 			model = MultiLstm(8, 32, lyrs, 1)
# 			loss_function = nn.L1Loss(size_average=True)
# 			optimizer = optim.Adam(model.parameters())
# 			model.zero_grad()

# 			with torch.no_grad():
# 				data = torch.load('train_X.pt')
# 				labels = torch.load('train_y.pt')
# 				val_x = torch.load('val_X.pt')
# 				val_y = torch.load('val_y.pt')

			
# 			for epoch in range(60):
# 				cumloss=0
# 				valloss=0
# 				model.zero_grad()
# 				optimizer.zero_grad()
# 				model.hidden = model.init_hidden()

# 				out = model(data)
# 				loss=loss_function(out, labels)
# 				loss.backward()
# 				optimizer.step()
# 				cumloss += float(loss.data)
# 				with torch.no_grad():
# 					out_val = model(val_x)
# 					loss_val = loss_function(out_val, val_y)
# 					valloss += float(loss_val.data)
# 				print('      Epoch:', epoch+1,'/', EPOCH, '| Training Loss:', cumloss, '| Validation Loss:', valloss)
# 			avg+=valloss
# 		avgs2.append(avg/10)



# 				# for i in range(data.shape[1]):
# 				# 	model.zero_grad()
# 				# 	optimizer.zero_grad()

# 				# 	model.hidden = model.init_hidden()

# 				# 	out = model(data[:,i,:].view(data.shape[0],1,data.shape[2]))

# 				# 	loss = loss_function(out, labels[:,i,:].view(labels.shape[0],1,labels.shape[2]))
# 				# 	loss.backward()
# 				# 	optimizer.step()
# 				# 	cumloss += float(loss.data)

# 				# print('Epoch:', epoch+1,'/', EPOCH, '| Epoch Loss:', cumloss)

# 			#with torch.no_grad(): 
# 				# print(model(val_x[:3,0,:].view(3,1,data.shape[2])))
# 				# print(val_y[:3,0,:].view(3,1,labels.shape[2]))
# 				# torch.save(model.state_dict(), 'gru.pt')
# 	print(avgs)
# 	print(avgs2)
# 	plt.plot(nums, avgs, 'b')
# 	plt.plot(num2, avgs2, 'r')
# 	plt.show()
# layerTest()

def tt(modelt, hidden, lyr, epochh):
	out=0
	for i in range(1):
		if modelt=='gru': model = MultiGRU(28,hidden,lyr,1)
		elif modelt=='lstm': model = MultiLstm(28,hidden,lyr,1)
		print("   Run:", i)
		loss_function = nn.L1Loss(size_average=True)
		optimizer = optim.Adam(model.parameters())
		model.zero_grad()

		with torch.no_grad():
			data = torch.load('apptrainx.pt').float()
			labels = torch.load('apptrainy.pt').float()
			val_x = torch.load('appvalx.pt').float()
			val_y = torch.load('appvaly.pt').float()
		
		for epoch in range(7000):
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
	return out


nums = [1,16,32,64,128]
avgs_gru = []
avgs_lstm = []
epoch = 300
for lyr in nums:
	print("GRU Layers:", lyr)
	avgs_gru.append(tt('gru', 32, lyr, epoch))
for lyr in nums:
	print("LSTM Layers:", lyr)
	avgs_lstm.append(tt('lstm', 32, lyr, epoch))
print('done', avgs_gru)
print('done', avgs_lstm)

glyr = avgs_gru.index(min(avgs_gru))
llyr = avgs_lstm.index(min(avgs_lstm))
nums = [1,16,32,64,128]
avgs_gru2 = []
avgs_lstm2 = []
for dim in nums:
	print("GRU Layers:", glyr)
	avgs_gru2.append(tt('gru', dim, glyr, epoch))
for lyr in nums:
	print("LSTM Layers:", llyr)
	avgs_lstm2.append(tt('lstm', dim, llyr, epoch))
print(avgs_gru2)
print(avgs_lstm2)


