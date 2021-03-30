import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess, preprocess_create_graph
import collections
import pyvis
from pyvis.network import Network
import argparse

# Get year
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2019)
parser.add_argument("--analyze", default="Author")
args = parser.parse_args()

if args.analyze:
    print('\nGenerating Graph \n')
else:
    raise ValueError("You must enter a parameter to analyze, --analyze Author/Rank/Position/Area")


df = pd.read_csv('../data/SCSE_Records.csv')

def get_netx(year):
    df = pd.read_csv('../data/SCSE_Records.csv')
    netx_graph = preprocess_create_graph(df, year)
    return netx_graph

def format_data(df):
    # pyvis can only accept str or int types
    df = preprocess(df, 2019)
    df['author-pid'] = df['author-pid'].astype(str)
    df['co-author-pid'] = df['co-author-pid'].astype(str)
    df = df.fillna(0)
    df['weight'] = df['weight'].astype(int)
    print(df.info)
    return df

def py_visualise(df, year, analyze_param, output_filename='graph.html',show_buttons=True,only_physics_buttons=True):
    networkx_graph = get_netx(year)
    format_data(df)
    pyvis_graph = Network(height='750px', width='1000px')

    # for each node and its attributes in the networkx graph
    # attributes: {'author', 'year', 'Position', 'Gender', 'Management', 'Area', 'top_venue_count'}
    for node, node_attrs in networkx_graph.nodes(data=True):
        pyvis_graph.add_node(node, **node_attrs, label=node_attrs['author'])
        print(node,node_attrs)


    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value'] = edge_attrs['weight']
        # add the edge
        pyvis_graph.add_edge(source, target, **edge_attrs)

    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()

    return pyvis_graph.show(output_filename)

py_visualise(df, year = args.year, output_filename='graph_output.html', analyze_param = args.analyze)
