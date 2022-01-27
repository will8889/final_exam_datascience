import networkx as nx
import matplotlib.pyplot as plt

G=nx.read_edgelist('text.csv',delimiter=",")
for e in G.edges():
    print(e)

nx.draw(G)
plt.show()