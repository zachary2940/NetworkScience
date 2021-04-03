import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_create_graph 
import collections
# https://networkx.org/documentation/stable/reference/algorithms/centrality.html

def plot_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)
    plt.show()



df = pd.read_csv('../data/SCSE_Records.csv')
for year in range(2000,2021):
    G = preprocess_create_graph(df,year)
    for k in G.edges:
        print(G.edges[k]['weight'])
        exit()
    G.name = year
    # plot_degree_distribution(G)
    print(nx.is_directed(G))
    print('Number of Isolates: ',nx.number_of_isolates(G))
    print(nx.info(G))
    print('Average no. of edges: ', len(G.edges)/len(G.nodes))
    print('Average Clustering coefficient: ',nx.average_clustering(G))
    # print('Diameter: ', nx.diameter(G))
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        print('Average shortest path length: ', nx.average_shortest_path_length(C))
    print()
