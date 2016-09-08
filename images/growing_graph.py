import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import networkx as nx

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

G = nx.Graph()

nx.draw(G, ax=ax)
node_number = 0

line = ax.plot()

def init():
	return line

# initialization function: plot the background of each frame
def animate(i):
    	global node_number
    	node_number += 1
	n = node_number
   	G.add_node(node_number)
    	G.add_edge(node_number, 0)
    	nx.draw(G)
	return line

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1, interval=100, blit=True)

# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('first_anim.mp4', fps=3, extra_args=['-vcodec', 'libx264'])

plt.show()
