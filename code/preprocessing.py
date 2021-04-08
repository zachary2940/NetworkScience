import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import requests
import shutil
import time

import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import os

from tqdm import tqdm
import numpy as np

def scrape_scse_xml(faculty_path, top_path, xml_path):
    '''
    Scrape XML files. Do note that this is not fool-proof and we will have
    to manually download/overwrite 14 of the 84 XML files (Tay Kian Boon does not have a DBLP link)
    '''
    f = pd.read_csv(faculty_path)
    t = pd.read_csv(top_path)

    # Convert "Software engg" to full term to match Top.csv later
    f.loc[~f['Area'].isin(list(t.Area)), 'Area'] = 'Software Engineering'
    # Save faculty.csv in an updated version.
    f = f.loc[:, ['Faculty', 'Position', 'Gender', 'Management', 'DBLP', 'Area']]
    f.sort_values('Area')
    f.to_csv('../data/Faculty.csv', index=False)

    # Begin preparing URLs for scraping in xml format. 
    urls = [url.replace('.html', '.xml') for url in list(f.DBLP)]


    for i, url in enumerate(urls):
        print(f.iloc[i].Faculty)
        print(url)
        print()
        
        filename = f.iloc[i].Faculty
        area = f.iloc[i].Area
        
        response = requests.get(url)
        with open(xml_path+'{}.xml'.format(filename), 'wb') as file:
            file.write(response.content)

def merge_faculty_csv_author_pid(faculty_path, xml_path):
    # Map Faculty names to DBLP's PID
    fac_df = pd.read_csv(faculty_path)
    fac_df['Faculty'] = fac_df['Faculty'].apply(lambda x: x.strip())
    fac_df['author-pid'] = [np.nan]*85

    # Update author-pid and confirm that the updates are correct
    for f in os.listdir(xml_path):
        cur_f = f.replace('.xml', '')
        try:
            tree = ET.parse(xml_path+f)
            root = tree.getroot()
            fac_df.loc[fac_df['Faculty']==cur_f, 'author-pid'] = root.attrib['pid']
        except:
            continue
    print("Number of rows with 'author-pid' filled up:", len(fac_df.loc[~fac_df['author-pid'].isnull()]))
    print("Number of rows without 'author-pid' filled up:", len(fac_df.loc[fac_df['author-pid'].isnull()]))
    try:
        fac_df = fac_df.drop(columns=['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'])
    except:
        pass
    fac_df.to_csv('../data/Faculty.csv', index=False)

def xml_to_csv(xml_path, csv_path):
    # With EDA all done, convert each faculty member's XML to CSV
    for f in tqdm(os.listdir(xml_path)):
        try:
            cols=['author','author-pid', 'paper', 'conference', 'year', 'title']
            rows = []
            tree = ET.parse(xml_path+f)
            root = tree.getroot()

            for node1 in root:
                paper = "" #node2.attrib['key']
                author = [] #node3.text
                author_pid = [] #node3.attrib['pid']
                conference = "" # node3.booktitle
                year = 0 # node3.year
                title = "" # node3.title
                if node1.tag == "r":
                    for node2 in node1:
                        paper = node2.attrib['key']
                        for node3 in node2:
                            if node3.tag == "author" or node3.tag == "editor":
                                author.append(node3.text)
                                author_pid.append(node3.attrib['pid'])
                            elif node3.tag == "booktitle":
                                conference = node3.text
                            elif node3.tag == "year":
                                year = node3.text
                            elif node3.tag == "title":
                                title = node3.text

                    for i in range(len(author)):
                        rows.append({
                            'author':author[i],
                            'author-pid':author_pid[i], 
                            'paper':paper, 
                            'conference':conference,     
                            'year':year, 
                            'title':title
                        })
        except:
            continue
        df = pd.DataFrame(rows, columns=cols)
        name = f.replace('.xml', '.csv')
        df.to_csv(csv_path+name, index=False)

def merge_scse_csv(csv_path):
    df = pd.DataFrame(columns=['author', 'author-pid', 'paper', 'conference', 'year', 'title'])
    expected_length = 0

    # Merge all 84 faculty CSVs into a single dataframe. Remove the csvs as they are no longer needed, using os.remove()
    for f in tqdm(os.listdir(csv_path)):
        if f == "Top.csv" or f=="Faculty.csv":
            continue
        df1 = pd.read_csv('../data/'+f)
        expected_length += len(df1)
        df = pd.concat([df, df1], ignore_index=True)

        # Clean up the unncessary csvs
        os.remove('../data/'+f)

    print("Successfully Completed?", expected_length==len(df))
    if expected_length==len(df):
        drop_scse_duplicates(df)
    else:
        print("Unsuccessful")


def drop_scse_duplicates(df):
    # Drop all duplicate rows, keep only the first occurence. There are multiple identical entries as a result of collaboration between
    # SCSE profs appearing in both of their CSV files.
    duplicates = df[df.duplicated()]
    print("Current length of df:", len(df))
    print("Number of Duplicated Rows (Excluding the first occurence):", len(duplicates))
    print("\nExpected length of df after removing duplicates:", len(df)-len(duplicates))
    df = df.drop_duplicates()
    duplicates = df[df.duplicated()]
    print("Length of df after removing duplicates:", len(df))
    print("Number of Duplicated Rows after Cleaning:", len(duplicates))

    fac_df = pd.read_csv('../data/Faculty.csv')
    fac_df = fac_df.drop(columns=['DBLP'])
    fac_author_pid = list(fac_df['author-pid'])
    joined_df = pd.merge(df, fac_df, on=['author-pid'], how='left')

    print("Number of SCSE Faculty Members in df:", len(df.loc[df['author-pid'].isin(fac_author_pid)]))
    print("Number of rows in joined_df successfully filled after join:", len(joined_df.loc[joined_df['Management'].notnull()]))
    partition_csvs(joined_df)

def partition_csvs(joined_df):
    # Create a Non_SCSE_df dataframe with only non-SCSE prof's entries, 
    # and SCSE_df with only SCSE prof's entries
    SCSE_df = joined_df.loc[joined_df['Position'].notnull()]
    Non_SCSE_df = joined_df.loc[~joined_df['Position'].notnull()]

    print("SCSE_df length:", len(SCSE_df))
    print("Non_SCSE_df length:", len(Non_SCSE_df))
    print("Successfully partitioned:", len(SCSE_df)+len(Non_SCSE_df)==len(joined_df))

    # Verify that the partitioning is successful
    print("Non_SCSE_df correctly partitioned:", len(Non_SCSE_df.loc[(Non_SCSE_df['Faculty'].notnull()) | (Non_SCSE_df['Position'].notnull()) |
                   (Non_SCSE_df['Gender'].notnull()) | (Non_SCSE_df['Management'].notnull()) |
                   (Non_SCSE_df['Area'].notnull())])==0)
    print("SCSE_df correctly partitioned:", len(SCSE_df.loc[(SCSE_df['Faculty'].isnull()) | (SCSE_df['Position'].isnull()) |
                   (SCSE_df['Gender'].isnull()) | (SCSE_df['Management'].isnull()) |
                   (SCSE_df['Area'].isnull())])==0)
    save_records(joined_df, Non_SCSE_df, SCSE_df, '../data/')

def save_records(joined_df, Non_SCSE_df, SCSE_df, data_path):
    # Save three files
    # 1. All_Records.csv contains SCSE and non-SCSE DBLP records.
    # 2. Non_SCSE_Records.csv contains only Non-SCSE DBLP records.
    # 3. SCSE_Records.csv contains only SCSE DBLP records
    # TL;DR All_Records = Non_SCSE_Records + SCSE_Records
    joined_df.to_csv(data_path+'All_Records.csv', index=False)
    Non_SCSE_df.to_csv(data_path+'Non_SCSE_Records.csv', index=False)
    SCSE_df.to_csv(data_path+'SCSE_Records.csv', index=False)
                 
                 
#scrape_scse_xml('../data/', '../data/', '../xml/')
#merge_faculty_csv_author_pid('../faculty/', '../xml/')
#xml_to_csv('../xml/', '../data/')
#merge_scse_csv('../data/')


def filter_year(df,year):
    df = df.loc[df['year']==year].reset_index(drop=True)
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

def tay_kian_boon_create_pid(df):
    df.loc[df['Faculty-co-author']=='Tay Kian Boon','co-author-pid']='solo'
    return df

def verify_correct_nodes(df):
    df_unique_pid = set(df['co-author-pid'].unique().tolist())
    df_faculty = pd.read_csv('../data/Faculty.csv')
    df_faculty.loc[df_faculty['Faculty']=='Tay Kian Boon','author-pid']='solo'

    df_faculty_unique_pid = set(df_faculty['author-pid'].unique().tolist())
    if len(df_faculty_unique_pid^df_unique_pid)!=0:
        raise Exception('Wrong no. of nodes: ',len(df_faculty_unique_pid^df_unique_pid)) 

def get_isolated_nodes(df):
    df_null = df[df.isnull().any(axis=1)].copy()
    df_null.loc[:,'year']=df['year'].unique()[0].copy()
    df_null.loc[:,'author']=df_null['Faculty-co-author'].copy()
    df_null.loc[:,'Position']=df_null['Position-co-author'].copy()
    df_null.loc[:,'Gender']=df_null['Gender-co-author'].copy()
    df_null.loc[:,'Management']=df_null['Management-co-author'].copy()
    df_null.loc[:,'Area']=df_null['Area-co-author'].copy()
    df_null.loc[:,'top_venue_count'] = 0

    df_null = df_null.set_index('co-author-pid')
    df_null = df_null.drop(['author-pid','weight', 'paper-list',
       'Faculty-co-author', 'Position-co-author', 'Gender-co-author',
       'Management-co-author', 'Area-co-author'],1)
    isolated_nodes_dict = df_null.to_dict('index')

    return isolated_nodes_dict

def create_graph(df):
    isolated_nodes_dict = get_isolated_nodes(df)
    df = df.dropna().reset_index(drop=True)
    G = nx.from_pandas_edgelist(df,source='author-pid',target='co-author-pid',edge_attr='weight')
    for node in isolated_nodes_dict:
        G.add_node(node)
    df_attributes = df.drop(['co-author-pid','weight', 'paper-list',
       'Faculty-co-author', 'Position-co-author', 'Gender-co-author',
       'Management-co-author', 'Area-co-author'],1)
    df_attributes = df_attributes.set_index('author-pid')
    df_attributes = df_attributes[~df_attributes.index.duplicated(keep='first')]
    author_attribute_dict = df_attributes.to_dict('index')
    nx.set_node_attributes(G, author_attribute_dict)
    nx.set_node_attributes(G, isolated_nodes_dict)
    # print("No of unique nodes:", len(G.nodes))
    # print("No of connections:", len(G.edges))
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

def preprocess(df,year):
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
    df = tay_kian_boon_create_pid(df)
    verify_correct_nodes(df)
    return df

'''
Inputs: 
    df - DataFrame
    year - int (2000 to 2020)

Outputs: 
    Networkx Graph
        Nodes containing attributes: 
        Edges containing attributes: weight

Example:
    df = pd.read_csv('../data/SCSE_Records.csv')
    G = preprocess_create_graph(df,2019)
'''


def preprocess_create_graph(df,year):
    df = preprocess(df,year)
    # df.to_csv('../data/graph.csv',index=False)
    G = create_graph(df)
    # visualize_graph(G) # just to check my work
    return G
