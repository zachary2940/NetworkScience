import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess
import collections
import pyvis
from pyvis.network import Network


def cr8_graph():
    net1 = Network(height='750px', width='100%')
    df = pd.read_csv('../data/SCSE_Records.csv')
    df = preprocess(df,2019)
    df['author-pid'] = df['author-pid'].astype(str)
    df['co-author-pid'] = df['co-author-pid'].astype(str)
    df = df.fillna(0)
    df['weight'] = df['weight'].astype(int)

    sources = df['author-pid']
    targets = df['co-author-pid']
    weights = df['weight']

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        net1.add_node(src, src, title=src)
        net1.add_node(dst, dst, title=dst)
        net1.add_edge(src, dst, value=w)

    neighbor_map = net1.get_adj_list()

    for node in net1.nodes:
        node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
        node['value'] = len(neighbor_map[node['id']])

    net1.show('gameofthrones.html')

cr8_graph()