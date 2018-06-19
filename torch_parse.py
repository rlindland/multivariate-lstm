import numpy as np
import torch

train_X	= torch.from_numpy(np.load('train_X.npy'))
train_y	= torch.from_numpy(np.load('train_y.npy'))
test_X	= torch.from_numpy(np.load('test_X.npy'))
test_y	= torch.from_numpy(np.load('test_y.npy'))

unbatched_data = train_X
unbatched_labels = train_y

for dset in [train_X, train_y, test_X, test_y]:
	temp_d = train_X.split(train_X.shape[0]//72, dim=0)
	if len(dset.shape)>1: dset = torch.cat(temp_d[:-1], dim=1)
	else: dset = torch.cat(temp_d[:-1], dim=1)
	print(dset.shape)

torch.save(train_X, 'train_X.pt')
torch.save(train_y, 'train_y.pt')
torch.save(test_X, 'test_X.pt')
torch.save(test_y, 'test_y.pt')