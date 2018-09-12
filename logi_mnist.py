import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy import optimize

from matplotlib import cm

#alpha = 0.00001

globm=0

fp_image=open('train-images.idx3-ubyte','rb')
fp_label=open('train-labels.idx1-ubyte','rb')

img = np.zeros((28,28))

s=fp_image.read(16)
l=fp_label.read(8)

X_list = []
Y_list = []

getsoos= np.zeros(10)

while True:
    s=fp_image.read(784)
    l=fp_label.read(1)
    if(not s):
        break
    if(not l):
        break
    img= np.reshape(struct.unpack(len(s)*'B', s),784)
    X_list.append(img)
    label = np.zeros(10)
    label[int(l[0])] = 1
    getsoos+=label
    Y_list.append(label)
    #lbl[index].append(img)
    globm=globm+1

X=np.reshape(X_list,(globm,784))
Y=np.reshape(Y_list, (globm,10))
X=X/128
newX = np.ones((globm,1))
X=np.append(newX,X,axis=1)
plt.imshow(np.reshape(img,(28,28)),cmap=cm.binary)
plt.show()
print(getsoos)
theta = np.zeros((785,10))
#편한 X, Y형태로 표현


def hypothesis(x):
    return 1/(1+np.exp(-x))


def costfunc(mytheta,x,y,m):
    return np.sum(-y*np.log(hypothesis(x@mytheta))-(1-y)* np.log(1-hypothesis(x@mytheta)))/m

#미분된 값
def grad(mytheta,x,y,m):
    return x.T@(hypothesis(x@mytheta)-y)/m
learn_m= globm*6//10
#learn: 60% CV 제외

X_learn = X[0:learn_m-1,:]
Y_learn = Y[0:learn_m-1,:]


mycost = lambda p: costfunc(p.reshape((785,10)),X_learn,Y_learn,learn_m)
mygrad = lambda p: grad(p.reshape((785,10)),X_learn,Y_learn,learn_m).reshape(785*10)

res=optimize.minimize(fun=mycost,x0=theta,jac=mygrad,method='CG',options={'disp':True, 'maxiter':100})
theta=res.x.reshape((785,10))
print("비용"+str(res.fun))

#0.5이상은 1로 판별하고 0.5 미만은 0으로 처리하여 결과를 뽑습니다, 판별단계

np.savetxt("whytheta.txt",theta)

X_test = X[learn_m:globm-1,:]
Y_test = Y[learn_m:globm-1,:]

pred= hypothesis(X_test@theta)

pred = np.where(pred<0.5,0,1)
correct = 0

for i in range(0,globm-learn_m-1):
    if(np.array_equal(Y_test[i], pred[i])):
        correct+=1

print("정확도"+str(100*correct/(globm-learn_m)))
