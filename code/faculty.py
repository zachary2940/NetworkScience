import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_create_graph 
from preprocessing import create_graph 
from preprocessing import preprocess_range 
from preprocessing import visualize_graph
from preprocessing import preprocess
from preprocessing import preprocess_authors
import collections
import math
from functools import reduce
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

def get_network_statistics(G,year):
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

def get_excellence_nodes(df,percentile=75):
    p = np.percentile(df['top_venue_count'].unique(), percentile) # return 50th percentile, e.g median.
    df_excellence = df.loc[df['top_venue_count']>p]
    return df_excellence[['author-pid','top_venue_count']].drop_duplicates(subset='author-pid')


def get_central_nodes(G):
    node_centrality_scores_degree = nx.degree_centrality(G)
    node_centrality_scores_eigenvector = nx.eigenvector_centrality(G)
    node_centrality_scores_betweeness = nx.betweenness_centrality(G)
    node_centrality_scores_closeness = nx.closeness_centrality(G)

    def get_top_k_central_nodes(node_centrality_scores):
        score_arr = []
        for key in node_centrality_scores:
            if node_centrality_scores[key] !=0:
                score_arr.append([key,node_centrality_scores[key]])
        sorted_score_arr = sorted(score_arr,key=lambda x:-x[1])
        return sorted_score_arr
    df_degree = pd.DataFrame(get_top_k_central_nodes(node_centrality_scores_degree),columns=['author-pid','degree_centrality'])
    df_eigenvector = pd.DataFrame(get_top_k_central_nodes(node_centrality_scores_eigenvector),columns=['author-pid','eigenvector_centrality'])
    df_betweeness = pd.DataFrame(get_top_k_central_nodes(node_centrality_scores_betweeness),columns=['author-pid','betweeness_centrality'])
    df_closeness = pd.DataFrame(get_top_k_central_nodes(node_centrality_scores_closeness),columns=['author-pid','closeness_centrality'])
    data_frames = [df_degree, df_eigenvector, df_betweeness,df_closeness]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['author-pid'],
                                            how='outer'), data_frames)
    return df_merged

def compare_excellence_centrality(df,percentile=75):
    excellence_nodes = get_excellence_nodes(df,percentile)
    G = create_graph(df)
    central_nodes = get_central_nodes(G)
    central_excellence_nodes = pd.merge(excellence_nodes,central_nodes,on=['author-pid'], how='outer')
    df_faculty = pd.read_csv('../data/Faculty.csv')
    central_excellence_nodes = pd.merge(central_excellence_nodes,df_faculty[['Faculty','author-pid']],on=['author-pid'], how='inner')
    print(central_excellence_nodes)
    central_excellence_nodes = central_excellence_nodes[central_excellence_nodes['top_venue_count'].notna()]
    overlap = central_excellence_nodes[~central_excellence_nodes[['degree_centrality','eigenvector_centrality','betweeness_centrality','closeness_centrality']].isnull().values.all(axis=1)]
    print('No. of central excellence nodes: ', len(overlap))
    central_excellence_nodes['degree_centrality'] = central_excellence_nodes['degree_centrality']*100
    central_excellence_nodes['eigenvector_centrality'] = central_excellence_nodes['eigenvector_centrality']*100
    central_excellence_nodes['betweeness_centrality'] = central_excellence_nodes['betweeness_centrality']*100
    central_excellence_nodes['closeness_centrality'] = central_excellence_nodes['closeness_centrality']*100
    central_excellence_nodes.plot()
    plt.show()

df = pd.read_csv('../data/SCSE_Records.csv')
year = 2011
G = preprocess_create_graph(df,year)
get_network_statistics(G,year)
df_collab = preprocess(df,year)
print(internal_collab(df_collab, 'Area'))
print(external_collab(df_collab, 'Area'))
print(external_collab(df_collab, 'Area',group='Computer Networks'))

df_authors=preprocess_authors(df,year,['l/BuSungLee','14/3737','1444536'])
G = create_graph(df_authors)
visualize_graph(G)

'''We define that a faculty is an excellence node if he/she has published in the top venue frequently (in the last 10 years or 
since his/her first publication if the first publication appears less than 10 years ago) in his/her respective area'''
df = preprocess_range(df,2010,2020)
compare_excellence_centrality(df)
