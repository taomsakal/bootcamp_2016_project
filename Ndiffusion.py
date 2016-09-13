import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as la
import pylab as pl

import animator as a

animator_obj = a.Animator()

nodes = 50
sick = 40  # Initial number of sick people
G = nx.erdos_renyi_graph(nodes, 0.4)
A = nx.to_numpy_matrix(G)
nx.draw_spring(G)
plt.style.use('ggplot')
plt.show()

print(A)

beta = 1  # The infection rate
gamma = 2  # The recovery rate. Set to zero to make this an SI model.

n = A.shape[1]

# make sure length of I_init is equal to number of nodes
# I_init = np.random.randint(2,size=nodes)


I_init = np.array([1] * (sick) + [0] * (nodes - sick))
S_init = 1 - I_init  # The sus. population.

# T = iterations of propagation to run
T = 100

# dt = size of time interval to simulate
dt = 0.01

eigenvalues, eigenvectors = la.eig(np.array(A))
index = np.argmax(eigenvalues)
vmax = eigenvectors[:index]

vmax = np.array([1] * nodes)

I = np.zeros((n, T))  # Nodes by trails
S = np.zeros((n, T))

I[:, 0] = I_init
S[:, 0] = S_init


# --- The main simulation --- # --------------------
def run_simulation(type, G, I_init, beta, gamma, T, dt):
    """
    Run the SIS model. If the recovery rate gamma is set to 0, then this is a SI model.
    :param type: The type of model to run. 'SI' or 'SIS' or 'SIR'.
    :param G: The networkx graph we are running the model over.
    :param I_init: A vector of the initial infected population. Todo: If pass in an integer n instead, then
                    make a vector where the first n nodes are infected.
    :param beta: The infection rate.
    :param gamma: The recovery rate.
    :param T: The number of iterations.
    :param dt: The size of the time interval to simulate.
    :return: None.
    """

    # Initialize some parameters
    A = nx.to_numpy_matrix(G)
    n = A.shape[1]  # The number of nodes

    # Generate infected population vector if needed.
    if isinstance(I_init, int):
        I_init = np.array([1] * (sick) + [0] * (nodes - sick))

    #S_init = 1 - I_init  # Susp. population.

    if type == 'SIS':
        run_SIS(G, I_init, beta, gamma, T, dt)
    elif type == 'SI':
        run_SIS(G, I_init, beta, 0, T, dt)
    elif type == 'SIR':
        run_SIR(G, I_init, beta, 0, T, dt)
    else:
        print("Invalid type for run_simulation()")


def run_SIS(G, I_init, beta, gamma, T, dt):
    """
    Run the SIS model. If the recovery rate gamma is 0, this is effectively the SI model.
    :params: See run_simulation.
    :return: None
    """
    for t in range(0, T - 1):
        for i in range(0, n - 1):
            sum1 = 0
            for j in range(0, n - 1):
                sum1 = sum1 + A[i, j] * I[j, t]

            recovered = gamma * I[i, t]
            I[i, t + 1] = max(0, min(1, (I[i, t] + dt * beta * S[i, t] * sum1 - recovered)))
            # S[i, t + 1] = max(0, min(1, S[i, t] - dt * beta * S[i, t] * sum1))
            S[i, t + 1] = 1 - I[i, t + 1]


def run_SIR(G, I_init, beta, gamma, T, dt):
    """
    Runs the SIR model.
    :params: See run_simulation
    :return: None
    """
    pass  # Todo.


run_simulation('SI', G, I_init, beta, gamma, T, dt)

S_w = np.dot(vmax, S)
I_w = np.dot(vmax, I)

# animator_obj.make_animation(G, I, T)

# --- Plotting --- # --------------------------

pl.figure(1)

# Ploting
# pl.plot(S, '-bs', label='Susceptibles')
pl.plot(S_w, '-bs', label='Susceptibles')

# pl.plot(I, '-ro', label='Infectious')
pl.plot(I_w, '-ro', label='Infectious')

# pl.legend(loc=0)
pl.title('SI epidemic without births or deaths')
pl.xlabel('Time')
pl.xlim([0, 50])  # Limit the x-axis
pl.ylabel('Susceptibles and Infectious')
# pl.savefig('2.5-SIS-high.png', dpi=900) # This does increase the resolution.
pl.show()
