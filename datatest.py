import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
from gru import MultiGRU
from lstm import MultiLstm
from rnn import MultiRNN
import matplotlib.pyplot as plt

np.random.seed(69)
torch.manual_seed(69)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.plot([1,2,3,4], [1,4,7,2], color='r', label='fuk')
# plt.legend()
# plt.show()

def tt(modelt, data, labels):
	outt=0
	for i in range(5):
		if modelt=='gru': model = MultiGRU(8,32,5,1)
		elif modelt=='lstm': model = MultiLstm(8,32,5,1)
		elif modelt=='rnn': model = MultiRNN(8,32,5,1)
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
		outt+=valloss
		print(valloss)
	print(outt)
	return outt/5

avgs_gru = []
avgs_lstm = []
avgs_rnn = []
counter = 1
for i in range(20):
	data=torch.load(str(i)+'x')
	labels=torch.load(str(i)+'y')
	print("RNN Layers:", counter) 
	avgs_rnn.append(tt('rnn', data, labels))
	counter+=1
counter = 1
for i in range(20):
	data=torch.load(str(i)+'x')
	labels=torch.load(str(i)+'y')
	print("GRU Layers:", counter) 
	avgs_gru.append(tt('gru', data, labels))
	counter+=1
counter = 1
for i in range(20):
	data=torch.load(str(i)+'x')
	labels=torch.load(str(i)+'y')
	print("lstm Layers:", counter) 
	avgs_lstm.append(tt('lstm', data, labels))
	counter+=1
print(avgs_gru)
print(avgs_lstm)
print(avgs_rnn)

xs = [i for i in range(1,401,20)]

plt.xlabel('Batches')
plt.ylabel('Validation MAE')
plt.title('Neural Net Performance')
plt.plot(xs, avgs_gru, color='g', label='GRU')
plt.plot(xs, avgs_lstm, color='b', label='LSTM')
plt.plot(xs, avgs_rnn, color='r', label='RNN')
plt.legend()
plt.show()