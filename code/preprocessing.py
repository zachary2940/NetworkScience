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

'''
[Pipeline1] EDA and Scraping XML Files
Scrape XML files using provided URL. Manually download some as not
all URLs having .html replaced with .xml will download the right file.
'''
f = pd.read_csv('../data/Faculty.csv')
t = pd.read_csv('../data/Top.csv')

# Convert "Software engg" to full term to match Top.csv later
f.loc[~f['Area'].isin(list(t.Area)), 'Area'] = 'Software Engineering'
# Save faculty.csv in an updated version.
f = f.loc[:, ['Faculty', 'Position', 'Gender', 'Management', 'DBLP', 'Area']]
f.sort_values('Area')
f.to_csv('../data/Faculty.csv', index=False)

# Begin preparing URLs for scraping in xml format. 
urls = [url.replace('.html', '.xml') for url in list(f.DBLP)]

# Scrap XML files. Do note that this is not fool-proof and we will have
# to manually download/overwrite 14 of the 85 XML files.
# Also, Tay Kian Boon is a staff in NTU, but the link given is for Tan Kian Boon. 
# There is no results for Tay Kian Boon in DBLP.
for i, url in enumerate(urls):
    print(f.iloc[i].Faculty)
    print(url)
    print()
    
    filename = f.iloc[i].Faculty
    area = f.iloc[i].Area
    
    response = requests.get(url)
    with open('../xml/{}.xml'.format(filename), 'wb') as file:
        file.write(response.content)


'''
[Pipeline2] Exploring XML and Determining Keys and Attributes Needed
Check out the tags in all XML files to determine what they mean, and which to retain in our database.
'''
# From our EDA, we can find that some of the xml files has a depth deeper than 3. 
# We will investigate the importance of the sub-nodes and perform some manually cleaning as we go along:

# This seeks to find how deep the trees go for every xml file. 
# Manual cleaning is done later to ensure that all important information is not removed when converting XML to CSV later
nodes = defaultdict(set)

for f in os.listdir('../xml/'):
    # There is a folder named "Problematic". Ignore it.
    if f == "Problematic":
        continue
    nodes[f] = defaultdict(set)
    # Dig how deep the tree goes. No need for dynamic programming as we know
    # from EDA it won't go beyond depth 4.
    tree = ET.parse('../xml/'+f)
    root = tree.getroot()
    for node1 in root:
        if node1.tag == "r":
            nodes[f]['node1'].add(node1.tag)
            for node2 in node1:
                nodes[f]['node2'].add(node2.tag)
                for node3 in node2:
                    nodes[f]['node3'].add(node3.tag)
                    for node4 in node3:
                        nodes[f]['node4'].add(node4.tag)
                        for node5 in node4:
                            nodes[f]['node5'].add(node5.tag)
        else: nodes[f]['node1'].add(node1.tag)


# We learn that tags beyond node 3 are not important. We removed them from the XML files as we explored.
# Before we begin exploring, we need to make sure that the keys in any node2 header are unique for every xml file.
# Understandably, when we do stack XML files there may be a chance of rows having the same key as a result of collaboration between NTU Profs.

# From the results we know the keys are all unique.
for f in os.listdir('../xml/'):
    try:
        tree = ET.parse('../xml/'+f)
        root = tree.getroot()
        n_elements = int(root.attrib['n'])
        n_found = set()
        for node1 in root:
            if node1.tag == "r":
                for node2 in node1:
                    n_found.add(node2.attrib['key'])
        if n_elements != len(n_found):
            print(f)
    except:
        continue

# We can confirm that keys in each xml file are unique, we next investigate whether it is worth considering nodes 
# that do not appear in all xml files (i.e. proceedings, incollections). To reference the project requirement,
# > Here we measure research collaboration as co-authorship among faculty members in **scientific papers/articles**
depth1 = []
depth2 = []
depth3 = []
for k in nodes.keys():
    depth1 += (list(nodes[k]['node1']))
    depth2 += (list(nodes[k]['node2']))
    depth3 += (list(nodes[k]['node3']))
    
depth1 = Counter(depth1)
depth2 = Counter(depth2)
depth3 = Counter(depth3)

# "Lets look for nodes which do not appear in all xml files. 
# From our EDA we know that the following professors have very few publications:
# 1. Tan Kheng Leong 
# 2. Loke Yuan Ren
# 3. Oh Hong Lye
# 4. Tay Kian Boon (0 results found on DBLP)
# As such, any nodes with >81 occurences are OK.")
print("\nNode 1:")
for k, v in depth1.items():
    print("{}: {}".format(k,v))
print()

print("Node 2:")
for k, v in depth2.items():
    print("{}: {}".format(k,v))
print()

print("Node 3:")
for k, v in depth3.items():
    print("{}: {}".format(k,v))
   
# find XML files with less frequent node 2 tags.
check = False
for k in nodes.keys():
    if 'book' in nodes[k]['node2']:
        check = True
        print("XML Files with <book>\t\t", k)
        
    if 'phdthesis' in nodes[k]['node2']:
        check = True
        print("XML Files with <phdthesis>\t", k)
    if check:
        print()
        check = False

# Find XML files with less frequent node 3 tags.
check = False
for k in nodes.keys():
    if 'school' in nodes[k]['node3']:
        check = True
        print("XML Files with <school>\t\t", k)
        
    if 'note' in nodes[k]['node3']:
        check = True
        print("XML Files with <note>\t\t", k)
        
    if 'cite' in nodes[k]['node3']:
        check = True
        print("XML Files with <cite>\t\t", k)
        
    if 'cdrom' in nodes[k]['node3']:
        check = True
        print("XML Files with <cdrom>\t\t", k)

    if check:
        print()
        check = False

'''
[Pipeline3] Exporting XML to CSV
With all key information identified we export the XML files to CSV, retaining all key information.
'''
# Map Faculty names to DBLP's PID
fac_df = pd.read_csv('../data/Faculty.csv')
fac_df['Faculty'] = fac_df['Faculty'].apply(lambda x: x.strip())
fac_df['author-pid'] = [np.nan]*85

# We can verify each xml file name is the same as the Faculty column in Faculty.csv.
# Note that Tay Kian Boon does not have an XML file.
fac = defaultdict(set)
found = 0
for i, f in enumerate(os.listdir('../xml/')):
    if (list(fac_df.loc[fac_df['Faculty']==f.replace('.xml', '')]['Faculty'])):
        found += 1
    else:
        print("Unmatched Files:", f.replace('.xml', ''))
print("Total matches found between .XML file names and fac_df:", found)

# Update author-pid and confirm that the updates are correct
for f in os.listdir('../xml/'):
    cur_f = f.replace('.xml', '')
    try:
        tree = ET.parse('../xml/'+f)
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

# With EDA all done, convert each faculty member's XML to CSV
for f in tqdm(os.listdir('../xml/')):
    try:
        cols=['author','author-pid', 'paper', 'conference', 'year', 'title']
        rows = []
        tree = ET.parse('../xml/'+f)
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
    df.to_csv("C:/Users/lowbe/Dropbox/CZ4071 Network Science/Project/data/"+name, index=False)
    
   
'''
[Pipeline4] Combining Every CSVs
We have 84 (excluding Tay Kian Boon) CSVs for each faculty member. Find a way to combine
all 84 CSVs along with Faculty.csv
'''
df = pd.DataFrame(columns=['author', 'author-pid', 'paper', 'conference', 'year', 'title'])
expected_length = 0

# Merge all 84 faculty CSVs into a single dataframe. Remove the csvs as they are no longer needed, using os.remove()
for f in tqdm(os.listdir('../data/')):
    if f == "Top.csv" or f=="Faculty.csv":
        continue
    df1 = pd.read_csv('../data/'+f)
    expected_length += len(df1)
    df = pd.concat([df, df1], ignore_index=True)
    
    # Clean up the unncessary csvs
    os.remove('../data/'+f)

print("Successfully Completed?", expected_length==len(df))


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
fac_df.head()

joined_df = pd.merge(df, fac_df, on=['author-pid'], how='left')
joined_df.head(4)

print("Number of SCSE Faculty Members in df:", len(df.loc[df['author-pid'].isin(fac_author_pid)]))
print("Number of rows in joined_df successfully filled after join:", len(joined_df.loc[joined_df['Management'].notnull()]))

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

# Save three files
# 1. All_Records.csv contains SCSE and non-SCSE DBLP records.
# 2. Non_SCSE_Records.csv contains only Non-SCSE DBLP records.
# 3. SCSE_Records.csv contains only SCSE DBLP records
# TL;DR All_Records = Non_SCSE_Records + SCSE_Records
joined_df.to_csv('../data/All_Records.csv', index=False)
Non_SCSE_df.to_csv('../data/Non_SCSE_Records.csv', index=False)
SCSE_df.to_csv('../data/SCSE_Records.csv', index=False)

'''
[Pipeline5] Final cleanup and Utilities for tackling the project
'''

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
    if len(df_faculty_unique_pid^df_unique_pid)==0:
        print('Correct no. of nodes')
    else:
        print('Wrong no. of nodes: ',len(df_faculty_unique_pid^df_unique_pid))

def get_isolated_nodes(df):
    df_null = df[df.isnull().any(axis=1)]
    df_null = df_null.set_index('co-author-pid')
    isolated_nodes_dict = df_null.to_dict('index')
    return isolated_nodes_dict

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
    tay_kian_boon_create_pid(df)
    verify_correct_nodes(df)
    isolated_nodes_dict = get_isolated_nodes(df)

    df = df.dropna().reset_index(drop=True)
    df.to_csv('../data/graph.csv',index=False)
    G = create_graph(df)
    for node in isolated_nodes_dict:
        G.add_node(node)
    visualize_graph(G) # just to check my work
    return G


df = pd.read_csv('../data/SCSE_Records.csv')
print(df.head())
G = preprocess_create_graph(df,2019)


# pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
# print(df)
