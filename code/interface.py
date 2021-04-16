import collections
import dash
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from colour import Color
from datetime import datetime
from textwrap import dedent as d
from preprocessing import preprocess, preprocess_create_graph, preprocess_range, preprocess_authors, create_graph
import plotly.express as px
import math
import dash_table
import faculty

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Graph Network"

EXCELLENCE_PERCENTILE = 50

# '83/6096', 'b/SSBhowmick', '33/885', '78/5155', '79/8116', '1444536', '126/4778', '14/3737'


def network_graph(year_range, option, authors=None):
    if option == '1000Nodes':
        df = pd.read_csv('../data/SCSE_top_1000_nodes_V3.csv')
        G = preprocess_create_graph(df, year_range)
    elif option == 'Excellence':
        df = pd.read_csv('../data/SCSE_Records.csv')
        df_range = preprocess_range(df, [2010, 2021])
        df_excellence = faculty.get_excellence_nodes(
            df_range, EXCELLENCE_PERCENTILE)
        excellence_pid_set = set(list(df_excellence['author-pid']))
        # print(excellence_pid_set)
    else:
        df = pd.read_csv('../data/SCSE_Records.csv')

    if authors != None:
        df_authors = preprocess_authors(df, year_range, authors)
        G = create_graph(df_authors)
    else:
        G = preprocess_create_graph(df, year_range)

    pos = nx.drawing.layout.spring_layout(G, k=0.35, iterations=30)

    if option == '1000Nodes':
        pos = nx.drawing.layout.spring_layout(G, k=0.65, iterations=50)

    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    traceRecode = []

    # edges scatter plot

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        weight = G.edges[edge]['weight']
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]), showlegend=False,
                           mode='lines', text="",
                           line={'width': weight},
                           marker=dict(color='grey'),
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        # hovertext in middle
        middle_node_trace = go.Scatter(
            x=tuple([(x0+x1)/2]),
            y=tuple([(y0+y1)/2]), showlegend=False,
            text="",
            hovertext=["Weight: " + str(weight)],
            mode='markers',
            hoverinfo='text',
            textfont=dict(
                family="sans serif",
                size=10,
                color="black"
            ),
            marker=dict(
                opacity=0
            )
        )
        # middle_node_trace['hovertext'] += tuple([hovertext])
        traceRecode.append(middle_node_trace)
        index = index + 1

    ###############################################################################################################################################################
    # nodes scatter plot
    colorsIdxPosition = {'Professor': 'mediumpurple', 'Associate Professor': 'lightcoral',
                         'Lecturer': 'gold', 'Senior Lecturer': 'limegreen', 'Assistant Professor': 'saddlebrown'}
    colorsIdxManagement = {'Y': 'blue', 'N': 'darkred'}
    colorsIdxGender = {'M': 'blue', 'F': 'darkred'}
    colorsIdxArea = {'Computer Networks': 'aquamarine', 'Computer Graphics': 'crimson', 'Computer Architecture': 'chocolate',
                     'AI/ML': 'darkblue', 'Cyber Security': 'lightgreen', 'HCI': 'magenta', 'Distributed Systems': 'tomato',
                     'Information Retrieval': 'gold', 'Data Management': 'darkgoldenrod', 'Data Mining': 'cyan', 'Computer Vision': 'black',
                     'Multimedia': 'saddlebrown', 'Software Engg': 'darkgrey', 'Bioinformatics': 'steelblue'}
    idxOption = {'Position': colorsIdxPosition, 'Management': colorsIdxManagement,
                 'Area': colorsIdxArea, 'Gender': colorsIdxGender}

    s = set([None, '1000Nodes', 'Original'])
    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        if 'author' not in G.nodes[node]:
            continue
        hovertext = "AuthorName: " + str(G.nodes[node]['author']) + "<br>" + "AuthorPid: " + str(G.nodes[node]['author-pid']) + "<br>" + "Position: " + str(
            G.nodes[node]['Position']) + "<br>" + "Mangement: " + str(
                G.nodes[node]['Management']) + "<br>" + "Area: " + str(
                G.nodes[node]['Area'])

        node_trace = go.Scatter(x=tuple([x]), y=tuple([y]), hovertext=tuple([hovertext]), text=tuple([G.nodes[node]['author']]),
                                legendgroup='', showlegend=False, mode='markers+text',
                                textposition="bottom center",
                                hoverinfo="text", marker={'size': 10, 'color': 'black'}, textfont=dict(
                                family="sans serif",
                                size=10,
                                color="black"
                                ))
        if option == 'Excellence':
            if G.nodes[node]["author-pid"] in excellence_pid_set:
                node_trace['legendgroup'] = 'Excellence Nodes'
                node_trace['marker']['color'] = 'gold'
        elif option not in s:
            node_trace['legendgroup'] = G.nodes[node][option]
            node_trace['marker']['color'] = idxOption[option][G.nodes[node][option]]

        index = index + 1
        traceRecode.append(node_trace)

    if option == 'Excellence':
        node_trace = go.Scatter(x=tuple([None]), y=tuple([None]),
                                legendgroup='Excellence Nodes', showlegend=True, mode='markers', name='Excellence Nodes',
                                marker={'size': 10, 'color': 'gold'})
        traceRecode.append(node_trace)
    elif option not in s:
        for k in idxOption[option]:
            node_trace = go.Scatter(x=tuple([None]), y=tuple([None]),
                                    legendgroup=k, showlegend=True, mode='markers', name=k,
                                    marker={'size': 10, 'color': idxOption[option][k]})
            traceRecode.append(node_trace)
    #####################################################################################################################
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Interactive Visualization', showlegend=True, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False,
                                   'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False,
                                   'showticklabels': False},
                            height=1200,
                            clickmode='event+select'
                            )}
    return figure


def display_network_statistics(year_range, option=None, authors=None):
    if option == '1000Nodes':
        df = pd.read_csv('../data/SCSE_top_1000_nodes_V3.csv')
        G = preprocess_create_graph(df, year_range)
    elif authors != None:
        df = pd.read_csv('../data/SCSE_Records.csv')
        df_collab = preprocess_authors(df, year_range, authors)
        G = create_graph(df_collab)
    else:
        df = pd.read_csv('../data/SCSE_Records.csv')
        G = preprocess_create_graph(df, year_range)
    return faculty.get_network_statistics(G, year_range)


def display_network_collaboration(year_range, category, authors=None):
    df = pd.read_csv('../data/SCSE_Records.csv')
    if authors:
        df_collab = preprocess_authors(df, year_range, authors)
    else:
        df_collab = preprocess_range(df, year_range)
    df_collab = df_collab.dropna()
    df_collab = df_collab.loc[df_collab.index.repeat(df_collab.weight)]
    if category == 'authors':
        df_collab = df_collab[['author', 'Faculty-co-author']]
    else:
        df_collab = df_collab[[category, category+'-co-author']]

    df_collab.columns = ['Groups', 'Groups_']
    # fig = px.density_heatmap(df_collab, x="Groups", y="Groups_")
    fig = px.density_heatmap(df_collab, x="Groups", y="Groups_").update_xaxes(
        categoryorder='category ascending').update_yaxes(categoryorder='category ascending')
    # fig = px.density_heatmap(df_collab, x="Groups", y="Groups_").update_xaxes(
    #     categoryorder="total descending").update_yaxes(categoryorder="total descending")

    return fig


def display_degree_distribution(year_range, option=None):
    if option == '1000Nodes':
        df = pd.read_csv('../data/SCSE_top_1000_nodes_V3.csv')
        G = preprocess_create_graph(df, year_range)
    else:
        df = pd.read_csv('../data/SCSE_Records.csv')
        G = preprocess_create_graph(df, year_range)
    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    logDegreeCount = {}
    deg = []
    cnt = []
    for k in degreeCount:
        cnt.append(math.log(degreeCount[k], 10))
        if k == 0:
            deg.append(k)
        else:
            deg.append(math.log(k, 10))
    df = pd.DataFrame({'degree': deg, 'count': cnt})
    return px.scatter(df, x="degree", y="count", trendline="ols")


def display_excellence_central_corr(option=None):
    if option == '1000Nodes':
        df = pd.read_csv('../data/SCSE_top_1000_nodes_V3.csv')
    else:
        df = pd.read_csv('../data/SCSE_Records.csv')
    df = preprocess_range(df, [2010, 2021])
    df_excellence_central = faculty.compare_excellence_centrality(
        df, percentile=EXCELLENCE_PERCENTILE)
    df_excellence_central = df_excellence_central.fillna(0)
    df_corr = df_excellence_central.corr()
    fig = px.imshow(df_corr)
    return fig


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

tab_style = {
    "background": "#323130",
    'text-transform': 'capitalize',
    'color': 'white',
    'border': 'grey',
    'font-size': '11px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding': '12px'
}

tab_selected_style = {
    "background": "grey",
    'text-transform': 'capitalize',
    'color': 'white',
    'font-size': '11px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding': '12px'
}

app.layout = html.Div([


    html.Div(
        className="row",
        children=[
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **Select Year Range To Visualize**\n
                            Slide the bars to define the year range chosen.
                            """)),
                    html.Div(
                        className="year-slider",
                        children=[
                            dcc.RangeSlider(
                                id='year-range-slider',
                                min=2000,
                                max=2021,
                                step=1,
                                value=[2018, 2019],
                                allowCross=False,
                                vertical=True,
                                verticalHeight=400,
                                marks={
                                    2000: {'label': '2000'},
                                    2001: {'label': '2001'},
                                    2002: {'label': '2002'},
                                    2003: {'label': '2003'},
                                    2004: {'label': '2004'},
                                    2005: {'label': '2005'},
                                    2006: {'label': '2006'},
                                    2007: {'label': '2007'},
                                    2008: {'label': '2008'},
                                    2009: {'label': '2009'},
                                    2010: {'label': '2010'},
                                    2011: {'label': '2011'},
                                    2012: {'label': '2012'},
                                    2013: {'label': '2013'},
                                    2014: {'label': '2014'},
                                    2015: {'label': '2015'},
                                    2016: {'label': '2016'},
                                    2017: {'label': '2017'},
                                    2018: {'label': '2018'},
                                    2019: {'label': '2019'},
                                    2020: {'label': '2020'},
                                    2021: {'label': '2021'}
                                }
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '545px', 'margin-left': '10px'}
                    ),
                    html.Div(className="twelve columns",
                             children=[
                                 dcc.Tabs(id="tabs-styled-with-inline", value=None, vertical=True, children=[
                                     dcc.Tab(label='SCSE Original', value='Original', style=tab_style,
                                             selected_style=tab_selected_style),
                                     dcc.Tab(label='Position', value='Position', style=tab_style,
                                             selected_style=tab_selected_style),
                                     dcc.Tab(label='Management', value='Management', style=tab_style,
                                             selected_style=tab_selected_style),
                                     dcc.Tab(label='Area', value='Area', style=tab_style,
                                             selected_style=tab_selected_style),
                                     dcc.Tab(label='Excellence Nodes', value='Excellence', style=tab_style,
                                             selected_style=tab_selected_style),
                                     dcc.Tab(label='Add 1000 Nodes', value='1000Nodes', style=tab_style,
                                             selected_style=tab_selected_style),
                                 ], style={'height': '300px', 'width': '200px'}),
                                 html.Div(id='tabs-content-inline')
                             ]),
                    html.Div(className="twelve columns",
                             children=[
                                 dcc.Markdown(
                                     d("""**Log-Log Degree Distribution**""")),
                                 dcc.Graph(
                                     id="degree_dist", figure=display_degree_distribution([2018, 2019])),
                             ], style={'height': '400px', 'width': '350px'})
                ]
            ),
            html.Center(
                className="eight columns",
                children=[html.Div([
                    html.Div([html.H2("SCSE Network Graph")],
                             className="row",
                             style={'textAlign': "center"}),
                    dcc.Markdown(d("""**Search by author pid**""")),
                    dcc.Markdown(d("""Enter in comma seperated author pid""")),
                    dcc.Markdown(
                        d("""E.g. Excellence nodes: 126/4778, 1444536, 83/6096, 79/8116, 33/885, 78/5155, b/SSBhowmick, 14/3737""")),
                    html.Div(dcc.Input(id='input-on-submit', type='text',
                                       placeholder="76/440,47/2026-7,b/SSBhowmick",
                                                   style={
                                                       'width': '450px'
                                                   })),
                    html.Button('Submit', id='submit-val', n_clicks=0),
                    html.Div(id='container-button-basic',
                             children='Enter a comma seperated list of author-pid and press submit')
                ]),
                    html.Div([
                        dcc.Graph(id="my-graph",
                                  figure=network_graph([2018, 2019], None)),
                        html.H3("Scroll around for more visualizations")], style={'height': '1400px', 'width': '1000px'})
                ],
                style={'height': '1400px', 'width': '1000px'}
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Network Statistics**
                            """)),
                            dash_table.DataTable(
                                id='table_network_statistics',
                                columns=[{"name": i, "id": i}
                                         for i in display_network_statistics([2018, 2019]).columns],
                                data=display_network_statistics(
                                    [2018, 2019]).to_dict('records'),
                            ),
                        ],
                        style={'height': '400px', 'width': '300px'}),
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Choose collaboration category**
                            """)),
                            dcc.Dropdown(
                                id='collab-dropdown',
                                options=[
                                    {'label': 'Area', 'value': 'Area'},
                                    {'label': 'Management', 'value': 'Management'},
                                    {'label': 'Position', 'value': 'Position'},
                                    {'label': 'Authors', 'value': 'authors'}
                                ],
                                value='Area'
                            ),
                        ],
                        style={'height': '80px', 'width': '500px'}),
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Collaboration**
                            """)),
                            dcc.Graph(id="2d_hist",
                                      figure=display_network_collaboration([2018, 2019], 'Area')),
                        ],
                        style={'height': '500px', 'width': '500px'}),
                    html.Div(
                        className='two columns',
                        children=[
                            dcc.Markdown(
                                d("""**Excellence-Centrality Corelation**""")),
                            dcc.Graph(id="ec_corr",
                                      figure=display_excellence_central_corr()),
                        ],
                        style={'height': '500px', 'width': '500px'})
                ]
            )
        ]
    )
])

# callback for left side components


@ app.callback(
    [dash.dependencies.Output('my-graph', 'figure'),
     dash.dependencies.Output('container-button-basic', 'children')],
    [dash.dependencies.Input('submit-val', 'n_clicks'),
     dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('tabs-styled-with-inline', 'value')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, year_range, option, author_pid_str):
    found = 'not found'
    message = 'The input value was "{}" and the authors are {}'.format(
        author_pid_str, found)
    if author_pid_str == None:
        return network_graph(year_range, option), message
    author_pid_list = author_pid_str.replace(" ", "").split(",")
    df = pd.read_csv('../data/SCSE_Records.csv')
    df = preprocess_authors(df, year_range, author_pid_list)
    if len(df) != 0:
        found = 'found'
        return network_graph(year_range, option, authors=author_pid_list), 'The input value was "{}" and the authors are {}'.format(author_pid_str, found)
    else:
        return network_graph(year_range, option), message


# callback for right side components


@ app.callback(
    [dash.dependencies.Output('table_network_statistics', 'columns'),
     dash.dependencies.Output('table_network_statistics', 'data')],
    [dash.dependencies.Input('submit-val', 'n_clicks'),
     dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('tabs-styled-with-inline', 'value')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_network_statistics(n_clicks, year_range, option, author_pid_str):
    if author_pid_str == None:
        return [{"name": i, "id": i} for i in display_network_statistics(year_range, option).columns], display_network_statistics(year_range, option).to_dict('records')
    author_pid_list = author_pid_str.replace(" ", "").split(",")
    df = pd.read_csv('../data/SCSE_Records.csv')
    df = preprocess_authors(df, year_range, author_pid_list)
    if len(df) != 0:
        return [{"name": i, "id": i} for i in display_network_statistics(year_range, option, author_pid_list).columns], display_network_statistics(year_range, option, author_pid_list).to_dict('records')
    else:
        return [{"name": i, "id": i} for i in display_network_statistics(year_range, option).columns], display_network_statistics(year_range, option).to_dict('records')
    return [{"name": i, "id": i} for i in display_network_statistics(year_range, option).columns], display_network_statistics(year_range, option).to_dict('records')


@ app.callback(
    dash.dependencies.Output('degree_dist', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('tabs-styled-with-inline', 'value')])
def update_degree_dist(year_range, option):
    return display_degree_distribution(year_range, option)


@ app.callback(
    dash.dependencies.Output('ec_corr', 'figure'),
    [dash.dependencies.Input('tabs-styled-with-inline', 'value')])
def update_ec_corr(option):
    return display_excellence_central_corr(option)


@ app.callback(
    dash.dependencies.Output('2d_hist', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('collab-dropdown', 'value'),
     dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_network_collaboration(year_range, category, n_clicks, author_pid_str):
    if author_pid_str == None:
        return display_network_collaboration(year_range, category)
    author_pid_list = author_pid_str.replace(" ", "").split(",")
    df = pd.read_csv('../data/SCSE_Records.csv')
    df = preprocess_authors(df, year_range, author_pid_list)
    if len(df) != 0:
        return display_network_collaboration(year_range, category, author_pid_list)
    else:
        return display_network_collaboration(year_range, category)

if __name__ == '__main__':
    app.run_server(debug=True)
