import numpy as np
import torch

sets = []

dset = torch.from_numpy(np.load('app_trainx.npy'))

temp_d = dset.split(72, dim=0) #should be (72)
out = torch.cat(temp_d[:-1], dim=1)
# print(out.shape)
sets.append(out)

dset = dset[1:]
temp_d = dset.split(72, dim=0)
out = torch.cat(temp_d[:-1], dim=1)
#print(out.shape)
sets.append(out)

torch.save(sets[0], 'app_train_x.pt')
torch.save(sets[1], 'app_train_y.pt')

x = sets[0]
y = sets[1]

for batch in range(y.shape[1]-1):
	for row in range(y.shape[0]):
		if batch==120 and row==71: pass
		elif row == y.shape[0]-1: 
			if y[row, batch, 0]!=x[0, batch+1, 0]: print(row, batch)
		elif y[row, batch, 0]!=x[row+1, batch, 0]: print(row, batch)

vals = []

for dset in sets:
	val = dset[:,0,:].view(72,-1,28)
	dset = dset[:,1:,:]
	print(val.shape)
	print(dset.shape)
	for i in range(26,sets[0].shape[1]-1,26):
		val = torch.cat((val.view(72,-1,28),dset[:,i,:].view(72,-1,28)), dim=1)
		dset = torch.cat((dset[:,:i,:].view(72,-1,28),dset[:,i+1:,:].view(72,-1,28)), dim=1)
	print(val.shape)
	print(dset.shape)
	vals.append(dset)
	vals.append(val)

for i in range(0,2):
	print('')
	x=vals[i]
	y=vals[i+2]
	for batch in range(y.shape[1]-1):
		for row in range(y.shape[0]):
			if batch==120 and row==71: pass
			elif row == y.shape[0]-1: 
				if y[row, batch, 0]!=x[0, batch+1, 0]: print('f',row, batch)
			elif y[row, batch, 0]!=x[row+1, batch, 0]: print(row, batch)

torch.save(vals[0], 'apptrainx.pt')
torch.save(vals[1], 'appvalx.pt')
torch.save(vals[2], 'apptrainy.pt')
torch.save(vals[3], 'appvaly.pt')

