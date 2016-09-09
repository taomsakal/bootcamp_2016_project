
import scipy.integrate as spi
import numpy as np
import pylab as pl
import networkx as nx
from random import random
from numpy import linalg as la
from random import random 
import matplotlib

import scipy.io as sio


nodes = 20
G = nx.watts_strogatz_graph(nodes,3,0.5)
A = nx.adjacency_matrix(G)

#print(A.todense())

beta=1
gamma=2

n=A.shape[1]

#generate initial health of population (0 = uninfected)
I_init=np.asarray([0]*nodes)
I_init[0]=1

S_init = 1 - I_init
=======
#make sure length of I_init is equal to number of nodes
I_init = np.random.randint(2,size=nodes)
S_init = 1-I_init


#T = iterations of propagation to run
T=100

#dt = size of time interval to simulate
dt=1

#eigenvalues, eigenvectors = la.eig(np.asarray(A))
#index=np.argmax(eigenvalues)
#vmax= eigenvectors[:index]

vmax = np.asarray([1]*nodes)

I=np.zeros((n, T))
S=np.zeros((n, T))

I[:, 0] = I_init
S[:, 0] = S_init

#the ODE
for t in range(0, T-2):
    for i in range(0, n-1):
    	sum1=0;
        for j in range(0, n-1):
        	sum1 = sum1 + A[i, j]*I[j,t];

	I[i,t+1] = I[i,t] + dt*beta*S[i,t]*sum1;
    	S[i,t+1] = 1 - I[i,t+1];

S_w= np.dot(vmax,S)
I_w= np.dot(vmax,I)

pl.figure(1)

#Ploting
pl.plot(S, '-bs', label='Susceptibles')
#pl.plot(S_w, '-bs', label='Susceptibles')

pl.plot(I, '-ro', label='Infectious')
#pl.plot(I_w, '-ro', label='Infectious')

#pl.legend(loc=0)
pl.title('SI epidemic without births or deaths')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
#pl.savefig('2.5-SIS-high.png', dpi=900) # This does increase the resolution.
pl.show()

