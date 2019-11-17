import numpy as np
import matplotlib.pyplot as plt


def readFile(file):
    data = np.loadtxt(file,delimiter=',', dtype=float)
    n = len(data)
    return data[:,0:2].T, data[:,2], n

def f((ax,thres,label),data):
    return label*((data[ax]>thres)*2-1)

def cntErr(y,z,w):
    return np.sum((y!=z)*w)

def get_h(X,y,w):
    minErr = 999
    for ax in [0,1]:
        for i in range(len(X[0])+1):
            thres = -3 if i==0 else X[ax,i-1]
            err = cntErr(f((ax,thres,1),X),y,w)
            label = 1 if (err<0.5) else -1
            err = err if (err<0.5) else 1.0-err
            if err<minErr:
                minErr = err
                Ax,Thres,Label = ax,thres,label
    beta = np.log((1-minErr)/minErr)/2
    return (Ax,Thres,Label),beta,minErr

def updateWeights(X,y,w,h,beta):
    w = w*np.exp(-beta*y*f(h,X))
    w = w/np.sum(w)
    return w

def classify(X,H,Beta):
    T = len(H)
    try:
        n=len(X[0])
    except:
        n=1
    sum = np.zeros(n)
    for t in range(T):
        sum= sum+Beta[t]*f(H[t],X)
    return (sum>0)*2-1

def main():
    N = 400
    Accu = np.zeros(N,dtype='float32')

    X,y,length = readFile("train_adaboost.csv")
    Xt,yt,lengtht = readFile("test_adaboost.csv")

    H,Beta = [],[]
    w = np.ones(length)/length
    for i in range(N):
        h,beta,err=get_h(X,y,w)
        w = updateWeights(X,y,w,h,beta)
        H.append(h)
        Beta.append(beta)
        res = classify(X,H,Beta)
        rest= classify(Xt,H,Beta)
        Accu[i] = 1-1.0*np.sum(rest!=yt)/lengtht

    plt.plot(np.array(range(N)),Accu,label='')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Test Accuracy')
    plt.title('')
    plt.show()


main()
