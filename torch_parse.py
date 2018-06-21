import numpy as np
import torch

train_X	= torch.from_numpy(np.load('train_X.npy'))
train_y	= torch.from_numpy(np.load('train_y.npy'))
test_X	= torch.from_numpy(np.load('test_X.npy'))
test_y	= torch.from_numpy(np.load('test_y.npy'))

unbatched_data = train_X
unbatched_labels = train_y

sets = []

for dset in [train_X, test_X]:
	temp_d = dset.split(72, dim=0) #should be (72)
	out = torch.cat(temp_d[:-1], dim=1)
	print(out.shape)
	sets.append(out)

	dset = dset[1:]
	temp_d = dset.split(72, dim=0)
	out = torch.cat(temp_d[:-1], dim=1)
	print(out.shape)
	sets.append(out)

torch.save(sets[0], 'train_X.pt')
torch.save(sets[1], 'train_y.pt')
torch.save(sets[2], 'test_X.pt')
torch.save(sets[3], 'test_y.pt')

x = sets[0]
y = sets[1]

for batch in range(y.shape[1]):
	for row in range(y.shape[0]):
		if batch==120 and row==71: pass
		elif row == y.shape[0]-1: 
			if y[row, batch, 0]!=x[0, batch+1, 0]: print(row, batch)
		elif y[row, batch, 0]!=x[row+1, batch, 0]: print(row, batch)