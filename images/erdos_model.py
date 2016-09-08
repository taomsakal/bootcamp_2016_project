import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from array import array
import random

num = 20
G = nx.erdos_renyi_graph(num,0.2)
fig = plt.figure(figsize=(8,8))
pos=nx.shell_layout(G)

healthy = range(num)
infected = []

nodes = nx.draw_networkx_nodes(G,pos,nodelist=healthy,node_color='b')
#nodes = nx.draw_networkx_nodes(G,pos,nodelist=infected,node_color='g')
edges = nx.draw_networkx_edges(G,pos) 

def update(n):

	if n%10==0 and healthy:
		rand = random.choice(healthy)
		healthy.remove(rand)
		infected.append(rand)
	
#	if n%20==0 and infected:
#		heal = random.choice(infected)
#		healthy.append(heal)
#		infected.remove(heal)	
	
	nodes = nx.draw_networkx_nodes(G,pos,nodelist=healthy,node_color='w')
	nodes = nx.draw_networkx_nodes(G,pos,nodelist=infected,node_color='g')
  	return nodes,

anim = FuncAnimation(fig, update, interval=50, blit=True)

#plt.show()

anim.save('erdos_anim.mp4', fps=6, extra_args=['-vcodec', 'libx264'])
