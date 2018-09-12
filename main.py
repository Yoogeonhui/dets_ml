
import numpy as np
import matplotlib.pyplot as plt

iteration=150000
alpha=0.001
a=np.loadtxt("ex2data1.txt")


input('데이터를 시각화합니다.')



print(a.shape)
#def plotdata():
#    plt.plot(a.T[0], a.T[1], 'ro')
#    plt.axis([4, 25, -5, 25])
#    plt.ylabel('y')
#    plt.xlabel('x')
#plotdata()
#plt.show()



m=a.shape[0]
X = np.array([np.ones(m),a.T[0],a.T[1]])
Y = a.T[2]
#X (2*97) Y (1*97)
theta= np.zeros((3,1))


def plotdata():
    yes_list = []
    no_list= []
    for i in range(0, m):
        if (Y[i]== 1):
            yes_list.append(X.T[i])
        else :
            no_list.append(X.T[i])
    yes_array = np.reshape(yes_list,(len(yes_list),3))
    no_array = np.reshape(no_list, (len(no_list),3))
    yes_array = yes_array.T
    no_array = no_array.T
    plt.plot(yes_array[1],yes_array[2],'ro')
    plt.plot(no_array[1],no_array[2],'go')
    plt.axis([0, 100, 0, 100])
    plt.ylabel('y')
    plt.xlabel('x')

plotdata()
plt.show()
#X (3,100)
def hypothesis(theta2,x):
    return 1/(1+np.exp(-(theta2.T@x).T))

print(hypothesis(theta,X).shape)
#hypothesis 전치1*97 => 97*1

Y=np.reshape(Y,(m,1))
print(X.shape)
#경사감소법
for i in range (iteration):
    if i%100==0:
        print(str(i)+"번째 cost: "+str(np.sum(-Y*np.log(hypothesis(theta,X))-(1-Y)*np.log(1-hypothesis(theta,X)))/m))
    mysum=alpha*(X@(hypothesis(theta,X)-Y))/m
    theta=theta-mysum
    if i%100 == 0:
        print(theta)

correct = 0
pred= hypothesis(theta, X)
print(pred)

pred = np.where(pred<0.5,0,1)

for i in range(0,m):
    if(np.array_equal(Y[i], pred[i])):
        correct+=1
plotdata()
hypx = np.arange(0,100,0.1)
hypy = (-1/theta[2])*(theta[0]+theta[1]*hypx)
plt.plot(hypx,hypy)
plt.show()
print(100*correct/m)
