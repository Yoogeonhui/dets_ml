import numpy as np
import matplotlib.pyplot as plt

iteration=1500
alpha=0.01

input('데이터를 시각화합니다.')
a=np.loadtxt("ex1data1.txt")

def plotdata():
    plt.plot(a.T[0], a.T[1], 'ro')
    plt.axis([4, 25, -5, 25])
    plt.ylabel('y')
    plt.xlabel('x')
plotdata()
plt.show()

m=a.shape[0]
X = np.array([np.ones(m),a.T[0]])
Y = a.T[1]
theta= np.zeros((2,1))

def hypothesis(theta2,x):
    return np.matmul(theta2.T,x).T


Y=np.reshape(Y,(m,1))
print(Y.shape)

#경사감소법
for i in range (iteration):
    if i%100==0:
        print(str(i)+"번째 cost: "+str(np.sum(np.square(hypothesis(theta,X)-Y))/(2*m)))
    mysum=alpha*np.matmul((hypothesis(theta,X)-Y).T,X.T).T/m
    theta-=mysum

print(theta)
plotdata()
hypx = np.arange(4,25,0.1)
hypy = theta[0]+theta[1]*hypx
plt.plot(hypx,hypy)
plt.show()