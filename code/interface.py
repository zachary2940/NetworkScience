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

import faculty

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Graph Network"


def network_graph(year, option):
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
        if option != None and option != '1000Nodes' and option!='Original':
            node_trace['legendgroup'] = G.nodes[node][option]
            node_trace['marker']['color'] = idxOption[option][G.nodes[node][option]]

        index = index + 1
        traceRecode.append(node_trace)

    if option != None and option != '1000Nodes' and option!='Original':
        for k in idxOption[option]:
            node_trace = go.Scatter(x=tuple([None]), y=tuple([None]),
                                    legendgroup=k, showlegend=True, mode='markers', name = k,
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
                                     dcc.Tab(label='Original', value='Original', style=tab_style,
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
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(2019, None))],
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**\n
                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px','width':'500px'}),
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**\n
                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])

# callback for left side components


@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('year-range-slider', 'value'),
     dash.dependencies.Input('tabs-styled-with-inline', 'value')])
def update_output(year, option):
    return network_graph(year, option)

# callback for right side components
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('year-range-slider', 'value')])
def display_hover_data(year):
    df = pd.read_csv('../data/SCSE_Records.csv')
    G = preprocess_create_graph(df,year)
    return json.dumps(faculty.get_network_statistics(G,year), indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
