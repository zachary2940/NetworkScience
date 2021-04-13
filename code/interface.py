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
from preprocessing import preprocess, preprocess_create_graph
import plotly.express as px
import math
import dash_table
import faculty

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Graph Network"


def network_graph(year, option):
    if option == '1000Nodes':
        df = pd.read_csv('../data/SCSE_Records.csv')
        G = preprocess_create_graph(df, year)
    else:
        df = pd.read_csv('../data/SCSE_Records.csv')
        G = preprocess_create_graph(df, year)
    pos = nx.drawing.layout.spring_layout(G, k=0.35, iterations=50)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    traceRecode = []

    # edges scatter plot

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        weight = G.edges[edge]['weight']
        hovertext = "AuthorName: " + str(G.nodes[node]['author'])
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
    colorsIdxArea = {'Computer Networks': 'aquamarine', 'Computer Graphics': 'crimson', 'Computer Architecture': 'chocolate',
                     'AI/ML': 'darkblue', 'Cyber Security': 'lightgreen', 'HCI': 'magenta', 'Distributed Systems': 'tomato',
                     'Information Retrieval': 'gold', 'Data Management': 'darkgoldenrod', 'Data Mining': 'cyan', 'Computer Vision': 'black',
                     'Multimedia': 'saddlebrown', 'Software Engg': 'darkgrey', 'Bioinformatics': 'steelblue'}
    idxOption = {'Position': colorsIdxPosition, 'Management': colorsIdxManagement,
                 'Area': colorsIdxArea}

    col_list = []

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "AuthorName: " + str(G.nodes[node]['author']) + "<br>" + "Position: " + str(
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
        if option != None and option != '1000Nodes' and option != 'Original':
            node_trace['legendgroup'] = G.nodes[node][option]
            node_trace['marker']['color'] = idxOption[option][G.nodes[node][option]]

        index = index + 1
        traceRecode.append(node_trace)

    if option != None and option != '1000Nodes' and option != 'Original':
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
                            height=600,
                            clickmode='event+select'
                            )}
    return figure


def display_network_statistics(year):
    df = pd.read_csv('../data/SCSE_Records.csv')
    G = preprocess_create_graph(df, year)
    return faculty.get_network_statistics(G, year)


def display_network_collaboration(year, category):
    df = pd.read_csv('../data/SCSE_Records.csv')
    df_collab = preprocess(df, year)
    df_collab = df_collab.dropna()
    df_collab = df_collab.loc[df_collab.index.repeat(df_collab.weight)]
    df_collab = df_collab[[category, category+'-co-author']]
    df_collab.columns = ['Groups', 'Groups_']
    fig = px.density_heatmap(df_collab, x="Groups", y="Groups_").update_xaxes(
        categoryorder="total descending").update_yaxes(categoryorder="total descending")

    return fig


def display_degree_distribution(year):
    df = pd.read_csv('../data/SCSE_Records.csv')
    G = preprocess_create_graph(df, year)
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
    html.Div([html.H1("SCSE Network Graph")],
             className="row",
             style={'textAlign': "center"}),

    html.Div(
        className="row",
        children=[
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **Select Year To Visualize**\n
                            Slide the bar to define the year chosen.
                            """)),
                    html.Div(
                        className="year-slider",
                        children=[
                            dcc.Slider(
                                id='year-range-slider',
                                min=2000,
                                max=2020,
                                step=1,
                                value=2019,
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
                                    2020: {'label': '2020'}
                                }
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '445px', 'margin-left': '10px'}
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
                                     dcc.Tab(label='Add 1000 Nodes', value='1000Nodes', style=tab_style,
                                             selected_style=tab_selected_style),
                                 ], style={'height': '50px', 'width': '200px'}),
                                 html.Div(id='tabs-content-inline')
                             ])
                ]
            ),
            html.Center(
                className="eight columns",
                children=[dcc.Graph(id="my-graph", figure=network_graph(2019, None)),
                          html.P("Scroll down for more visualizations"),
                          html.Div(
                    className='twelve columns',
                    children=[
                        dcc.Markdown(d("""**Excellence Nodes vs Central Nodes**""")),
                        html.Pre(id='d', style=styles['pre']),
                        html.Div(
                            className='twelve columns',
                            children=[
                                dcc.Markdown(d("""
                            **Log-Log Degree Distribution**
                            """)),
                                dcc.Graph(id="degree_dist",
                                          figure=display_degree_distribution(2019))
                            ],
                            style={'height': '400px', 'width': '300px'})
                    ],
                    style={'height': '300px', 'width': '300px'})
                ],
                style={'height': '800px', 'width': '800px'}
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
                                         for i in display_network_statistics(2019).columns],
                                data=display_network_statistics(
                                    2019).to_dict('records'),
                            ),
                        ],
                        style={'height': '300px', 'width': '300px'}),

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
                                    {'label': 'Position', 'value': 'Position'}
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
                                      figure=display_network_collaboration(2019, 'Area')),
                        ],
                        style={'height': '300px', 'width': '500px'})
                ]
            )
        ]
    )
])

# callback for left side components
@ app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('tabs-styled-with-inline', 'value')])
def update_output(year, option):

    return network_graph(year, option)

# callback for right side components
@ app.callback(
    [dash.dependencies.Output('table_network_statistics', 'columns'),
     dash.dependencies.Output('table_network_statistics', 'data')],
    [dash.dependencies.Input('year-range-slider', 'value')])
def update_network_statistics(year):
    return [{"name": i, "id": i} for i in display_network_statistics(year).columns], display_network_statistics(year).to_dict('records')


@ app.callback(
    dash.dependencies.Output('degree_dist', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value')])
def display_click_data(year):
    return display_degree_distribution(year)


@ app.callback(
    dash.dependencies.Output('2d_hist', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('collab-dropdown', 'value')])
def update_network_collaboration(year, category):
    return display_network_collaboration(year, category)


if __name__ == '__main__':
    app.run_server(debug=True)
