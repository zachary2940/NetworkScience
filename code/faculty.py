import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_create_graph, preprocess_core
from preprocessing import create_graph
from preprocessing import preprocess_range
from preprocessing import visualize_graph
from preprocessing import preprocess
from preprocessing import preprocess_authors
from preprocessing import load_top_venue_dict
from preprocessing import add_coauthor
from preprocessing import drop_author_self_link
from preprocessing import add_weight
from preprocessing import add_paper_list
from preprocessing import drop_author_coauthor_duplicates
import collections
import math
from functools import reduce
# https://networkx.org/documentation/stable/reference/algorithms/centrality.html


def plot_degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
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
    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    logDegreeCount = {}
    for k in degreeCount:
        if k == 0:
            logDegreeCount[k] = math.log(degreeCount[k], 10)
        else:
            logDegreeCount[math.log(k)] = math.log(degreeCount[k], 10)
    deg, cnt = zip(*logDegreeCount.items())
    fig, ax = plt.subplots()
    plt.scatter(deg, cnt)

    plt.title("Degree Distribution log-log plot")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(['{}'.format(round(d, 3)) for d in deg])
    plt.show()


def experiment():
    G = nx.path_graph(4)
    nx.add_path(G, [10, 11, 12])
    visualize_graph(G)
    print(nx.is_connected(G))  # by default strongly connected


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
        'Position': {'Associate Professor', 'Professor', 'Assistant Professor', 'Senior Lecturer', 'Lecturer'},
        'Gender': {'M', 'F'},
        'Management': {'Y', 'N'},
        'Area': {'AI/ML',
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
        collab_dict[g] = sum(df.loc[(df[column] == g) & (
            df[collaboration_column] == g)].weight)
    if normalize:
        collab_dict = normalize_values(collab_dict)

    return collab_dict


def external_collab(df, column, *, group=False, normalize=False):
    # Find the external collaboration of every group in a given column.
    comparison_dict = {
        'Position': {'Associate Professor', 'Professor', 'Assistant Professor', 'Senior Lecturer', 'Lecturer'},
        'Gender': {'M', 'F'},
        'Management': {'Y', 'N'},
        'Area': {'AI/ML',
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
            collab_dict[g] = sum(df.loc[(df[column] == group) & (
                df[collaboration_column] == g)].weight)
        # We are finding collaboration from each group to all other groups
        else:
            collab_dict[g] = sum(df.loc[(df[column] == g) & (
                ~(df[collaboration_column] == g))].weight)

    if normalize:
        collab_dict = normalize_values(collab_dict)

    return collab_dict


def get_network_statistics(G, year):
    G.name = year
    n_connected_components = 1
    for C in (G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len)):
        largest_C = C
        n_connected_components -= 1
    info = {'Number of Isolates': nx.number_of_isolates(G),
            'No. of nodes': float(nx.info(G).split('\n')[-3].split(':')[-1].strip()),
            'No. of edges': float(nx.info(G).split('\n')[-2].split(':')[-1].strip()),   
            'Average no. of edges': len(G.edges)/len(G.nodes),
            'Average degree': float(nx.info(G).split('\n')[-1].split(':')[-1].strip()),
            'Average Clustering coefficient': nx.average_clustering(G),
            'No. of nodes in largest connected component': len(largest_C.nodes),
            'Diameter of largest connected component': nx.diameter(largest_C),
            'Average shortest path length of LCC': nx.average_shortest_path_length(largest_C)
            }
    df_stats = pd.DataFrame.from_records([info]).T
    df_stats=df_stats.reset_index()
    df_stats.columns = ['Attributes','Statistics']
    return df_stats


def get_excellence_nodes(df, percentile=75):
    # return 50th percentile, e.g median.
    p = np.percentile(df['top_venue_count'].unique(), percentile)
    df_excellence = df.loc[df['top_venue_count'] > p]
    return df_excellence[['author-pid', 'top_venue_count']].drop_duplicates(subset='author-pid')


def get_central_nodes(G):
    node_centrality_scores_degree = nx.degree_centrality(G)
    node_centrality_scores_eigenvector = nx.eigenvector_centrality(G)
    node_centrality_scores_betweeness = nx.betweenness_centrality(G)
    node_centrality_scores_closeness = nx.closeness_centrality(G)
    def get_top_k_central_nodes(node_centrality_scores):
        score_arr = []
        for key in node_centrality_scores:
            if node_centrality_scores[key] != 0:
                score_arr.append([key, node_centrality_scores[key]])
        return score_arr
    df_degree = pd.DataFrame(get_top_k_central_nodes(
        node_centrality_scores_degree), columns=['author-pid', 'degree_centrality'])
    df_eigenvector = pd.DataFrame(get_top_k_central_nodes(
        node_centrality_scores_eigenvector), columns=['author-pid', 'eigenvector_centrality'])
    df_betweeness = pd.DataFrame(get_top_k_central_nodes(
        node_centrality_scores_betweeness), columns=['author-pid', 'betweeness_centrality'])
    df_closeness = pd.DataFrame(get_top_k_central_nodes(
        node_centrality_scores_closeness), columns=['author-pid', 'closeness_centrality'])
    data_frames = [df_degree, df_eigenvector, df_betweeness, df_closeness]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['author-pid'],
                                                    how='outer'), data_frames)
    return df_merged


def compare_excellence_centrality(df, percentile=75):
    excellence_nodes = get_excellence_nodes(df, percentile)
    G = create_graph(df)
    central_nodes = get_central_nodes(G)
    central_excellence_nodes = pd.merge(excellence_nodes, central_nodes, on=[
                                        'author-pid'], how='outer')
    df_faculty = pd.read_csv('../data/Faculty.csv')
    central_excellence_nodes_df = pd.merge(central_excellence_nodes, df_faculty[[
                                        'Faculty', 'author-pid']], on=['author-pid'], how='inner')
    return central_excellence_nodes_df

def faculty_member_collab(df, fac_set):
    '''
    Given a hashable fac_set containing author-pid, find the collaboration
    of each member of fac_set with every other member over the years.

    Returns a dictionary with keys being each member in fac_set, and value being another
    dictionary with key being the year and values being another dictionary
    with keys being the co-author-pid (who belongs in fac_set) and values being the weights
    '''
    collab_dict = defaultdict(lambda: defaultdict(dict))
    col = 'author-pid'
    col2 = 'co-author-pid'
    prep_df = preprocess_range(df, 2000, 2021)
    for member in fac_set:
        temp_df = prep_df.loc[(prep_df['author-pid'] == member)
                              & (prep_df['co-author-pid'].isin(fac_set))]

        temp_dict = temp_df[['year', 'weight',
                             'co-author-pid']].to_dict('list')
        for i in range(len(temp_df)):
            collab_dict[member][temp_dict['year'][i]].update(
                {temp_dict['co-author-pid'][i]: temp_dict['weight'][i]})

    return collab_dict

def trim_merged_df(df):
    df = df[['author', 'author-pid', 'year', 'paper', 'conference', 'title']]
    return df

def add_top_venues_count_non_SCSE(df, top_venue_set):
    def f(conf, top_venue_set):
        if conf in top_venue_set:
            return 1
        return 0
    
    df['top_venue'] = df.apply(lambda x: f(x.conference, top_venue_set),axis=1)
    df['top_venue_count'] = df.groupby('author-pid')['top_venue'].transform('sum')
    return df

def total_weight_non_SCSE(df):
    df['total_weight'] = df.groupby(['author-pid'])['weight'].transform('sum')
    return df

def verify_number_authors(df):
    class NonIsolatedNodeException(Exception):
        def __init__(self, value):
            self.value = value

    verification_df = pd.read_csv('../data/Non_SCSE_Records.csv')
    actual_unique_authors = set(verification_df['author-pid'])
    given_unique_authors = set(df['author-pid'])
    
    authors_not_in_df = actual_unique_authors - given_unique_authors

    papers = set()
    
    for authors in authors_not_in_df:
        papers.update(verification_df[verification_df['author-pid']==authors].paper.tolist())
    for p in papers:
        try:
            if len(verification_df[verification_df['paper']==p]) > 1:
                raise NonIsolatedNodeException(p)
        except NonIsolatedNodeException as e:
            print("The paper {} has co-authors existing in Non_SCSE_Records but is not accounted for in given DataFrame.".format(e.value))        
            
def merge_SCSE_non_SCSE(SCSE_path, non_SCSE_path):
    non_scse_df = pd.read_csv(non_SCSE_path)
    scse_df = pd.read_csv(SCSE_path)
    df = pd.concat([non_scse_df, scse_df]).reset_index(drop=True)
    return df

def create_non_SCSE_graph(df):   
    df = df.dropna().reset_index(drop=True)
    G = nx.from_pandas_edgelist(df,source='author-pid',target='co-author-pid',edge_attr='weight')        
    df_attributes = df.drop(['co-author-pid','weight', 'paper-list'],1)
    df_attributes = df_attributes.set_index('author-pid')
    df_attributes = df_attributes[~df_attributes.index.duplicated(keep='first')]
    author_attribute_dict = df_attributes.to_dict('index')
    nx.set_node_attributes(G, author_attribute_dict)

    return G


def drop_redundant_cols_non_SCSE(df):
    df = df.drop(['conference','top_venue','title'],axis=1).reset_index(drop=True)
    return df

def final_trim_non_SCSE(df):
    df = df[['author', 'author-pid', 'paper', 'year', 'top_venue_count', 'co-author-pid', 'total_weight', 'weight', 'closeness_centrality']]
    df = remove_SCSE(df)
    return df

def remove_SCSE(df):
    scse_df = pd.read_csv('../data/SCSE_Records.csv')
    scse_set = set(scse_df['author-pid'])
    df = df[~df['author-pid'].isin(scse_set)]
    return df

def normalized_centrality(df):
    maximum_degree = len(df)-1
    df['normalized_degree_centrality'] = df['total_weight'].apply(lambda x: x/maximum_degree)

    return df

def select_top_1000_non_SCSE_nodes(df, save_file=False, file_path='../data/top_1000_nodes.csv'):
    top_venue_author_list = list(df[df['top_venue_count']>0].sort_values(['top_venue_count', 'normalized_degree_centrality'], ascending=False)['author-pid'].unique())
    top_degree_centrality_author_list = list(df[(df['top_venue_count']==0) & (df['normalized_degree_centrality']>=df.normalized_degree_centrality.quantile(0.95))].sort_values(['normalized_degree_centrality'])['author-pid'].unique())
    top_closeness_centrality_author_list = list(df[((df['normalized_degree_centrality']<df.normalized_degree_centrality.quantile(0.95))) & (df['top_venue_count']==0)].sort_values(['closeness_centrality'])['author-pid'].unique())
    
    ret_df = pd.DataFrame(columns=['author', 'author-pid', 'paper', 'top_venue_count', 'co-author-pid', 'total_weight', 'weight', 'closeness_centrality', 'normalized_degree_centrality'])
    ret_df = pd.concat([ret_df, df[df['author-pid'].isin(top_venue_author_list)]])
    ret_df = pd.concat([ret_df, df[df['author-pid'].isin(top_degree_centrality_author_list[:(1000-len(top_venue_author_list))])]])
    ret_df = pd.concat([ret_df, df[df['author-pid'].isin(top_closeness_centrality_author_list[:(1000-len(ret_df['author-pid'].unique()))])]])
    
    if save_file:
        ret_df.to_csv('../data/top_1000_nodes_V3.csv', index=False, encoding='utf8')
    
    print("Number of Unique Authors:", len(ret_df['author-pid'].unique()))
    print("Number of Rows:", len(ret_df))
    print("Number of authors in top venue:", len(top_venue_author_list))
    print("Number of authors in top 95% degree centrality and not in top venue:", len(top_degree_centrality_author_list))
    print("Number of authors with highest closeness centrality but not in top venue and not in top 95% degree centrality:", 1000-len(top_degree_centrality_author_list)-len(top_venue_author_list))
    return ret_df

def select_non_SCSE(read_csv=True, save_file=False):
    df = merge_SCSE_non_SCSE('../data/SCSE_Records.csv', '../data/Non_SCSE_Records.csv')
    df = trim_merged_df(df)
    top_venue_dict = load_top_venue_dict()
    top_venue_set = set().union(*(value for value in top_venue_dict.values()))
    df = add_top_venues_count_non_SCSE(df,top_venue_set)
    df = add_coauthor(df)
    df = drop_author_self_link(df)
    df = add_weight(df)
    df = add_paper_list(df)
    df = drop_author_coauthor_duplicates(df)
    df = drop_redundant_cols_non_SCSE(df)
    df = total_weight_non_SCSE(df)

    if read_csv:
        central_nodes = pd.read_csv('../data/Non_SCSE_Centrality_V3.csv')

    else:
        G = create_non_SCSE_graph(df)
        # get_central_nodes takes about 15 mins to run.
        # The results have been stored in Non_SCSE_Centrality.csv for convenience
        central_nodes = get_central_nodes(G)
        central_nodes.to_csv('../data/Non_SCSE_Centrality_V3.csv', index=False)

    df = df.merge(central_nodes, on='author-pid', how='left')
    df = final_trim_non_SCSE(df)
    verify_number_authors(df)
    df = normalized_centrality(df)
    df = select_top_1000_non_SCSE_nodes(df, save_file)
    
    return df

# df_scse = pd.read_csv('../data/Non_SCSE_Records.csv')
# print(len(df_scse['author-pid']))
# df = pd.read_csv('../data/top_1000_nodes_V3.csv')
# df = df.drop_duplicates(subset='author-pid')
# print(len(df['author-pid']))
# df = df_scse[df_scse['author-pid'].isin(df['author-pid'])]
# print(len(df['author-pid']))
# print(len(df['author-pid'].unique()))
# print(df)
# df_scse = pd.read_csv('../data/SCSE_Records.csv')

# # df =pd.merge(df,df_scse,how='inner')
# df = pd.concat([df,df_scse])
# print(df)
# print(len(df['author-pid'].unique()))
# df.to_csv('../SCSE_top_1000_nodes_V3.csv')

# year = 2018
# G = preprocess_create_graph(df,year)
# print(get_network_statistics(G,year))

# df_authors=preprocess_authors(df,year,['l/BuSungLee','14/3737','1444536'])
# G = create_graph(df_authors)
# visualize_graph(G)

# '''We define that a faculty is an excellence node if he/she has published in the top venue frequently (in the last 10 years or
# since his/her first publication if the first publication appears less than 10 years ago) in his/her respective area'''
# df = pd.read_csv('../data/SCSE_Records.csv')
# df = preprocess_range(df,2010,2021)
# compare_excellence_centrality(df, percentile=0)


# df = pd.read_csv('../data/SCSE_top_1000_nodes_V3.csv')
# df = preprocess_core(df)
# G = create_graph(df)