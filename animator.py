import random
from array import array

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation


class Animator(object):

    def __init__(self):
        self.I = None
        self.G = None

    def make_animation(self, G, I, T):
        """
        Makes the final animation
        :param G: The graph we are working over.
        :param I: Matrix of infected nodes vs time.
        :param T: The number of trails.
        :return: None, but saves the animation.
        """

        self.I = I
        self.G = G

        num_nodes = nx.number_of_nodes(G)

        fig = plt.figure(figsize=(8, 8))
        pos = nx.shell_layout(G)

        healthy = range(num_nodes)
        infected = []

        drawing = nx.draw_networkx_nodes(G, pos, nodelist=healthy, node_color='b')
        # drawing = nx.draw_networkx_nodes(G,pos,nodelist=infected,node_color='g')
        edges = nx.draw_networkx_edges(G, pos)

        anim = FuncAnimation(fig, self.update, interval=T, blit=True)  # This calls update with increasing n.
        anim.save('erdos_anim.mp4', fps=6, extra_args=['-vcodec', 'libx264'])  # Saves the animation


    def update(self, n):
        """
        Draws a frame of the animation.
        :param n: The frame number we are on.
        :return: A single drawn frame.
        """
        pop = [row[n] for row in self.I]  # The population at time n

        # Get list of indices for the healthy and sick populations.
        healthy = [i for node, i in enumerate(pop) if node == 0]
        infected = [i for node, i in enumerate(pop) if node == 1]

        # Draws the graph.
        pos = nx.shell_layout(self.G)  # Gets the position of a layout for the drawing.
        drawing = nx.draw_networkx_nodes(self.G, pos, nodelist=healthy, node_color='w')  # Draw healthy nodes white
        drawing = nx.draw_networkx_nodes(self.G, pos, nodelist=infected, node_color='g')  # Draw infected nodes green
        return drawing,




