
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#np.set_printoptions(suppress=True)


''' Constant Declaration'''
eps_mean = 0
eps_var = 1


data = np.loadtxt('airfoil_self_noise.dat')
X = np.copy(data)
n,p = len(X),6
t = np.copy(X[:,5])
X[:,5] = np.ones(n)


def plotMLE(data,Wa):
  n,m = len(data),len(Wa)
  l = np.zeros(m,dtype='float32')
  for j in range(m):
    for i in range(n):
      l[j] += 0.0 - np.log(2*np.pi*eps_var) - (data[i,5]-Wa[j]*data[i,1]-eps_mean)**2/(2*eps_var)
  l_argmax = np.argmax(l)
  print Wa[l_argmax],l[l_argmax]

  plt.plot(Wa,l,'-',label='l(w)')
  plt.plot(Wa[l_argmax],l[l_argmax],'ro',label='estimated MLE w_a')
  plt.xlabel('w_a')
  plt.ylabel('Log-likelihood')
  plt.title('Log-likelihood w.r.t. w_a')
  #plt.legend(loc='bottom right')
  plt.show()


def logPosterior(X,t,w,mean,var):
  return ( -np.linalg.norm(np.matmul(X,w.transpose())-t)**2 - np.linalg.norm(w-mean)**2 ) /2  - 0.5*np.log(2*np.pi)*n - 0.5*np.log(2*np.pi*var)*p

def main():
  data = np.loadtxt('airfoil_self_noise.dat')

  ''' Q 5.1 '''
  Wa = np.arange(-5,20.01,0.01)
  #plotMLE(data,Wa)

  ''' Q 5.2 '''
  mean_set = [0 , 10 , 500]
  w = [0.0]*6

  for mean in mean_set:
    def func(w):
      return 0.0-logPosterior(X,t,w,mean,1)
    res = scipy.optimize.minimize(func, w, method="L-BFGS-B",options={'maxcor': 1000,'gtol': 1e-07,'maxfun': 1500000,'maxiter': 1500000})
    print 'Mean=',mean,'\n',res.x,'\n\n'

if __name__ == "__main__":
  main()
