import torch

# train = torch.load('train_X.pt')
# train1 = train[:,:30,:]; print(train1.shape); torch.save(train1, 'train1.pt')
# train2 = train[:,:60,:]; print(train2.shape); torch.save(train2, 'train2.pt')
# train3 = train[:,:90,:]; print(train3.shape); torch.save(train3, 'train3.pt')
# train4 = train[:,:120,:]; print(train4.shape); torch.save(train4, 'train4.pt')

# trainy = torch.load('train_y.pt')
# trainy1 = trainy[:,:30,:]; print(trainy1.shape); torch.save(trainy1, 'trainy1.pt')
# trainy2 = trainy[:,:60,:]; print(trainy2.shape); torch.save(trainy2, 'trainy2.pt')
# trainy3 = trainy[:,:90,:]; print(trainy3.shape); torch.save(trainy3, 'trainy3.pt')
# trainy4 = trainy[:,:120,:]; print(trainy4.shape); torch.save(trainy4, 'trainy4.pt')

# vx = torch.load('test_X.pt')
# vy = torch.load('test_y.pt')
# valx = vx[:,:1,:]
# valy = vy[:,:1,:]
# for i in range(50,486,50):
# 	valx=torch.cat((valx,vx[:,i,:].view(72,1,8)), dim=1)
# 	valy=torch.cat((valy,vy[:,i,:].view(72,1,8)), dim=1)
# torch.save(valx, 'valx.pt')
# torch.save(valy, 'valy.pt')
# print(valx.shape)
# print(valy.shape)

counter = 0
x = torch.load('apptrainx.pt')
y = torch.load('apptrainy.pt')
print(x[:,:2,:])
print(x.shape, y.shape)
for i in range(1,262,13):
	print(i)
	xx=x[:,:i,:]
	yy=y[:,:i,:]
	torch.save(xx, 'app'+str(counter)+'x')
	torch.save(yy, 'app'+str(counter)+'y')
	print(xx.shape)
	print(yy.shape)
	counter+=1

for i in range(21):
	print('')
	x=torch.load('app'+str(i)+'x')
	y=torch.load('app'+str(i)+'y')
	for batch in range(y.shape[1]-1):
		for row in range(y.shape[0]):
			if batch==120 and row==71: pass
			elif row == y.shape[0]-1: 
				if y[row, batch, 0]!=x[0, batch+1, 0]: print('f',row, batch)
			elif y[row, batch, 0]!=x[row+1, batch, 0]: print(row, batch)


# vx = torch.load('train_X.pt')
# vy = torch.load('train_y.pt')
# valx = vx[:,:1,:]
# valy = vy[:,:1,:]
# for i in range(20,121,20):
# 	valx=torch.cat((valx,vx[:,i,:].view(72,1,8)), dim=1)
# 	valy=torch.cat((valy,vy[:,i,:].view(72,1,8)), dim=1)
# torch.save(valx, 'valx.pt')
# torch.save(valy, 'valy.pt')
# print(valx.shape)
# print(valy.shape)
