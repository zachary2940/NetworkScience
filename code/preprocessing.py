import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def filter_year(df,year):
    df = df.loc[df['year']==year]
    return df

def load_top_venue_dict():
    df_top = pd.read_csv('../data/Top.csv')
    df_top = df_top.set_index('Area')
    top_venue_dict = df_top.to_dict('index')
    for k in top_venue_dict:
        top_venue_dict[k]=set([top_venue_dict[k]['Venue'].strip()])

    top_venue_dict['Data Management'].add('SIGMOD Conference')
    top_venue_dict['Data Mining'].update(['ACM SIGKDD','KDD'])
    top_venue_dict['Information Retrieval'].add('SIGIR')
    top_venue_dict['AI/ML'].add('NIPS')
    top_venue_dict['HCI'].add('CHI')
    top_venue_dict['Software Engineering'].add('ACM/IEEE ICSE')
    top_venue_dict['Software Engg'] = top_venue_dict['Software Engineering']
    top_venue_dict['Computer Graphics'].add('SIGGRAPH')
    top_venue_dict['Multimedia'].update(['ACM Multimedia', 'ACM-MM'])

    return top_venue_dict

def add_top_venues_count(df,top_venue_dict):
    def f(conf,area,top_venue_dict):
        if conf != None:
            venue = top_venue_dict[area]
            if conf in venue:
                return 1
        return 0

    df['top_venue'] = df.apply(lambda x: f(x.conference,x.Area,top_venue_dict),axis=1)
    df['top_venue_count'] = df.groupby('author-pid')['top_venue'].transform('sum')
    return df

def add_coauthor(df):
    df_author_pid_list = df.groupby('paper')['author-pid'].apply(list).to_frame()
    df = df.join(df_author_pid_list,on='paper',lsuffix='', rsuffix='-list', how='outer')
    df = df.explode('author-pid-list').reset_index(drop=True)
    df.rename({'author-pid-list': 'co-author-pid'}, axis=1, inplace=True)
    return df

def drop_author_self_link(df):
    df = df[df['author-pid']!=df['co-author-pid']].reset_index(drop=True)
    return df

def add_weight(df):
    df['weight'] = df.groupby(['author-pid', 'co-author-pid'])['author'].transform('count')# duplicates are removed later
    return df



def add_paper_list(df):
    df_paper_list = df.groupby(['author-pid', 'co-author-pid'])['paper'].apply(list).to_frame()
    df = df.join(df_paper_list,on=['author-pid', 'co-author-pid'],lsuffix='', rsuffix='-list', how='outer')
    return df

def drop_author_coauthor_duplicates(df):
    df = df.drop_duplicates(['author-pid', 'co-author-pid'],keep= 'last').reset_index(drop=True)
    return df

def add_coauthor_attributes(df):
    df_faculty = pd.read_csv('../data/Faculty.csv')
    df_faculty = df_faculty.set_index('author-pid')
    df = df.join(df_faculty, on='co-author-pid',lsuffix='', rsuffix='-co-author', how='outer')
    return df

def drop_redundant_cols(df):
    df = df.drop(['paper','conference','top_venue','Faculty','title','DBLP'],axis=1).reset_index(drop=True)
    return df

def create_graph(df):
    G = nx.from_pandas_edgelist(df,source='author-pid',target='co-author-pid',edge_attr='weight')
    print("No of unique nodes:", len(G.nodes))
    print("No of connections:", len(G.edges))
    return G

def visualize_graph(G):
    # all graph options
    graphs_viz_options = [nx.draw, nx.draw_networkx, nx.draw_circular, nx.draw_kamada_kawai, nx.draw_random, nx.draw_shell, nx.draw_spring]

    # plot graph option
    selected_graph_option = 0

    # plot
    plt.figure(figsize=(8,6), dpi=100) 
    graphs_viz_options[selected_graph_option](G)
    plt.show()

def preprocess_create_graph(df,year):
    df = filter_year(df,year)
    top_venue_dict = load_top_venue_dict()
    df = add_top_venues_count(df,top_venue_dict)
    df = add_coauthor(df)
    df = drop_author_self_link(df)
    df = add_weight(df)
    df = add_paper_list(df)
    df = drop_author_coauthor_duplicates(df)
    df = add_coauthor_attributes(df)
    df = drop_redundant_cols(df)
    print(df)

    # df.loc[df['Faculty-co-author']=='Tay Kian Boon']['co-author-pid']='solo'
    # print(df.loc[df['Faculty-co-author']=='Tay Kian Boon']['co-author-pid'])
    # exit()
    df_unique_pid = set(df['co-author-pid'].unique().tolist())
    df_faculty = pd.read_csv('../data/Faculty.csv')
    df_faculty_unique_pid = set(df_faculty['author-pid'].unique().tolist())
    print(df_unique_pid)
    print(df_faculty_unique_pid)
    print(df_faculty_unique_pid^df_unique_pid)

    df.to_csv('../data/graph.csv',index=False)
    G = create_graph(df)
    visualize_graph(G) # just to check my work
    return G

df = pd.read_csv('../data/SCSE_Records.csv')
print(df.head())
G = preprocess_create_graph(df,2019)


# pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
# print(df)