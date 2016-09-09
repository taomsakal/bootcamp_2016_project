import networkx as nx
import numpy as np
import pylab as pl
import numpy.linalg as la
import matplotlib.pyplot as plt
import animator as a

animator_obj = a.Animator()


nodes = 15
sick = 5  # Initial number of sick people

#G = nx.erdos_renyi_graph(nodes, 0.4)

#G = nx.watts_strogatz_graph(nodes,3, 0.5)


G = nx.erdos_renyi_graph(nodes, 0.7)


A = nx.to_numpy_matrix(G)

beta = 1  # The infection rate
gamma = 2

n = A.shape[1]

# make sure length of I_init is equal to number of nodes
# I_init = np.random.randint(2,size=nodes)

I_init = np.array([1] * (sick) + [0] * (nodes - sick))

S_init = 1 - I_init

# T = iterations of propagation to run
T = 100

# dt = size of time interval to simulate
dt = 0.01


eigenvalues, eigenvectors = la.eig(np.array(A))
index = np.argmax(eigenvalues)
vmax = eigenvectors[:index]

vmax = np.array([1] * nodes)

I = np.zeros((n, T))
S = np.zeros((n, T))

I[:, 0] = I_init
S[:, 0] = S_init

# the ODE
for t in range(0, T - 1):
    for i in range(0, n - 1):
        sum1 = 0
        for j in range(0, n - 1):
            sum1 = sum1 + A[i, j] * I[j, t]

        I[i, t + 1] = max(0, min(1, I[i, t] + dt * beta * S[i, t] * sum1))
        S[i, t + 1] = max(0, min(1, S[i, t] - dt * beta * S[i, t] * sum1))

# import sys
# sys.exit()

S_w = np.dot(vmax, S)
I_w = np.dot(vmax, I)

animator_obj.make_animation(G, I, T)

# --- Plotting --- # --------------------------

pl.figure(1)

# Ploting
# pl.plot(S, '-bs', label='Susceptibles')
#pl.plot(S_w, '-bs', label='Susceptibles')

# pl.plot(I, '-ro', label='Infectious')
#pl.plot(I_w, '-ro', label='Infectious')

# pl.legend(loc=0)
pl.title('SI epidemic without births or deaths')
pl.xlabel('Time')
pl.xlim([0,50])  # Limit the x-axis
pl.ylabel('Susceptibles and Infectious')
# pl.savefig('2.5-SIS-high.png', dpi=900) # This does increase the resolution.
#pl.show()
