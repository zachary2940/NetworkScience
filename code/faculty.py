import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_create_graph 
from preprocessing import visualize_graph
import collections
import math
# https://networkx.org/documentation/stable/reference/algorithms/centrality.html

def plot_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Distribution log-log plot")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

def plot_log_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    logDegreeCount = {}
    for k in degreeCount:
        if k==0:
            logDegreeCount[k]=math.log(degreeCount[k],10)
        else:
            logDegreeCount[math.log(k)]=math.log(degreeCount[k],10)
    deg, cnt = zip(*logDegreeCount.items())
    fig, ax = plt.subplots()
    plt.scatter(deg, cnt)

    plt.title("Degree Distribution log-log plot")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(['{}'.format(round(d,3)) for d in deg])
    plt.show()

def experiment():
    G = nx.path_graph(4)
    nx.add_path(G, [10, 11, 12])
    visualize_graph(G)
    print(nx.is_connected(G)) # by default strongly connected
    
def normalize_values(dictionary):
    '''
    Used in internal_collab() and external_collab()
    Normalize values based on ratio to total count
    '''
    total = sum(dictionary.values())
    for k, v in dictionary.items():
        dictionary[k] = v/total
    return dictionary

def internal_collab(df, column, *, normalize=False):
    # Find the internal collaboration of every group in a given column.
    comparison_dict = {
        'Position':{'Associate Professor', 'Professor', 'Assistant Professor', 'Senior Lecturer', 'Lecturer'},
        'Gender':{'M', 'F'},
        'Management':{'Y', 'N'},
        'Area':{'AI/ML',
                 'Bioinformatics',
                 'Computer Architecture',
                 'Computer Graphics',
                 'Computer Networks',
                 'Computer Vision',
                 'Cyber Security',
                 'Data Management',
                 'Data Mining',
                 'Distributed Systems',
                 'HCI',
                 'Information Retrieval',
                 'Multimedia',
                 'Software Engg'}
    }    
    # lets say we are comparing Area we will be looking at columns ['Area', 'Area-co-author']
    collaboration_column = column+'-co-author'
    
    collab_dict = {}
    
    for g in comparison_dict[column]:
        collab_dict[g] = sum(df.loc[(df[column]==g) & (df[collaboration_column]==g)].weight)
    if normalize:
        collab_dict = normalize_values(collab_dict)
        
    return collab_dict

def external_collab(df, column, *, group=False, normalize=False):
    # Find the external collaboration of every group in a given column.
    comparison_dict = {
        'Position':{'Associate Professor', 'Professor', 'Assistant Professor', 'Senior Lecturer', 'Lecturer'},
        'Gender':{'M', 'F'},
        'Management':{'Y', 'N'},
        'Area':{'AI/ML',
                 'Bioinformatics',
                 'Computer Architecture',
                 'Computer Graphics',
                 'Computer Networks',
                 'Computer Vision',
                 'Cyber Security',
                 'Data Management',
                 'Data Mining',
                 'Distributed Systems',
                 'HCI',
                 'Information Retrieval',
                 'Multimedia',
                 'Software Engg'}
    }    
    # If there's a group provided, verify that the group is in the column provided.
    if group:
        if group not in comparison_dict[column]:
            return "Provided group does not exist in provided column."
    
    # lets say we are comparing Area we will be looking at columns ['Area', 'Area-co-author']
    collaboration_column = column+'-co-author'
    collab_dict = {}
    
    for g in comparison_dict[column]:
        # We are finding collaboration from given group (parameter) to all other groups
        if group:
            # We don't want to find self-collaboration. Only external.
            if g == group:
                continue
            collab_dict[g] = sum(df.loc[(df[column]==group) & (df[collaboration_column]==g)].weight)
        # We are finding collaboration from each group to all other groups
        else:
            collab_dict[g] = sum(df.loc[(df[column]==g) & (~(df[collaboration_column]==g))].weight)
        
    if normalize:
        collab_dict = normalize_values(collab_dict)
        
    return collab_dict

df = pd.read_csv('../data/SCSE_Records.csv')
for year in range(2000,2021):
    G = preprocess_create_graph(df,year)
    # for k in G.edges:
    #     print(G.edges[k]['weight'])
    G.name = year
    # plot_degree_distribution(G)
    plot_log_degree_distribution(G)
    print('Number of Isolates: ',nx.number_of_isolates(G))
    print(nx.info(G))
    print('Average no. of edges: ', len(G.edges)/len(G.nodes))
    print('Average Clustering coefficient: ',nx.average_clustering(G))
    n_connected_components = 1
    print()
    print('Largest Connected Component')
    for C in (G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)):
        if n_connected_components==0:
            break
        n_connected_components-=1
        print('No. of nodes in connected component: ',len(C.nodes))
        print('Diameter of connected component: ', nx.diameter(C))
        print('Average shortest path length: ', nx.average_shortest_path_length(C))
    print()
