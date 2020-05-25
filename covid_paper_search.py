import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pickle
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

app = dash.Dash(suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

covid_papers = pd.read_csv("data/covid_papers.csv").drop(columns=['Unnamed: 0'])

vec = pickle.load(open("model/feature.pkl", "rb"))
lda = pickle.load(open("model/lda_model.pkl", "rb"))

options = [{'label': f"{title}", 'value':f"{index}"} for index, title in enumerate(covid_papers['title'])]

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.H1('Accelerate Your Research', style = {'text-align': 'center', 'font-family':'sans-serif'}),
    html.Br(),
    html.H5("Search the COVID-19 research database or enter the text from a paper you're interested in. The algorithm will read the paper and provide recommendations for further reading based on your interests.",
            style = {'text-align': 'center', 'font-family':'sans-serif'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Search Database', value='tab-1'),
        dcc.Tab(label='Enter Your Own Paper', value='tab-2'),
    ], style = {'width':'50%', 'margin':'auto'}),
    html.Br(),
    html.Div(id='tab-output'),
    html.Div(html.P(["Data: COVID-19 Open Research Dataset (CORD-19). 2020."], style= {'text-align': 'center','font-size': '10px', 'verticleAlign': 'text-bottom'})),
    html.Div(html.P(["Created by: Brian VandenAkker"], style= {'text-align': 'center','font-size': '10px', 'verticleAlign': 'text-bottom'})),
])

@app.callback(
    Output("paper-search", "options"),
    [Input("paper-search", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [paper for paper in options if search_value in paper["label"]]


@app.callback(Output('tab-output', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
                html.Div([dcc.Dropdown(id="paper-search", placeholder = "Search Papers in COVID-19 Open Research Dataset...")], style = {'width':'50%' , 'margin':'auto'}),
                dcc.Loading(html.Div(html.Ul(id = 'similar-papers-db'), style={'font-family':'sans-serif', 'width': '70%', 'margin':'10%'}), type = "cube"),
                dbc.Button("", id='button', outline=True, color="primary", className="mr-1", style={'margin-left': '-100%'})])
    elif tab == 'tab-2':
        return html.Div([dcc.Textarea(
                id='text-box',
                placeholder = "Enter the text you're interested in here. We'll process the text and return the papers from the COVID-19 database which are similar to your entry.",
                style={'width': '100%', 'height': 300}),
                dbc.Button("Process Text", id='button_load', outline=True, color="primary", className="mr-1", style={'margin-left': '46%',
                                                                                                                'margin-bottom': '10%px',
                                                                                                                'verticalAlign': 'middle'}),
                dcc.Loading(html.Div(html.Ul(id='similar-papers-new'), style={'font-family':'sans-serif', 'width': '90%', 'margin-right':'10%'}), type = "cube"),
                dbc.Button("Load More", id='button', outline=True, color="primary", className="mr-1", style={'margin-left': '-100%'})], style = {'margin-left':'10%','margin-right':'10%'})
@app.callback(
    Output(component_id='similar-papers-db', component_property= 'children'),
    [Input(component_id='paper-search', component_property='value'),
     Input(component_id='button', component_property = 'n_clicks')]
)
def similar_text_from_db(paper_search, n_clicks, n_articles = 10):
    if paper_search is None:
        raise PreventUpdate
    store_vals = list()
    topic_dist = [float(i) for i in covid_papers.loc[int(paper_search), 'topic_dist'].strip('[]').split(', ')]

    for i in range(len(covid_papers)):
        if(i in covid_papers.index):
            store_vals.append((int(i), (sqrt(mean_squared_error(topic_dist, [float(i) for i in covid_papers.loc[int(i), 'topic_dist'].strip('[]').split(', ')])))))
    most_similar = sorted(store_vals, key=itemgetter(1))
    if n_clicks is None:
        data = [covid_papers.loc[int(i[0]), ['title', 'url', 'abstract', 'authors' ,'publish_time']] for i in most_similar[1:(n_articles+1)]]
    else:
        data = [covid_papers.loc[int(i[0]), ['title', 'url', 'abstract', 'authors' ,'publish_time']] for i in most_similar[1:(n_articles+1 +(n_articles*n_clicks))]]
    out = []
    for info in data:
        out.append(dbc.ListGroup(
        [
            dbc.ListGroupItem(
                [
                    dbc.ListGroupItem(dbc.ListGroupItemHeading(info[0]), href=info[1], target="_blank"),
                    html.Br(),
                    dbc.ListGroupItemText(f"Authors: {info[3]}", style = {'text-align': 'left', 'font-size': 'small'}),
                    dbc.ListGroupItemText(f"Published: {info[4]}", style={'text-align': 'left', 'font-size': 'smaller'}),
                    dbc.ListGroupItemText(info[2], style={'text-align':'justify'}),
                ])], flush=True))
    out.append(html.Br())
    out.append(dbc.Button("Load More", id='button', outline=True, color="primary", className="mr-1", style={'margin-left': '46%',
                                                                                                    'margin-bottom': '10%px',
                                                                                                    'verticalAlign': 'middle'}))
    return out

@app.callback(
    Output(component_id='similar-papers-new', component_property= 'children'),
    [Input(component_id='button', component_property = 'n_clicks'),
    Input(component_id='button_load', component_property='n_clicks')],
    [State(component_id='text-box', component_property='value')]
)
def similar_text_new(n_clicks, load_page, process_text, n_articles = 10):
    if process_text is None:
        raise PreventUpdate

    vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("model/feature.pkl", "rb")))
    new_vec = vec.transform([process_text])
    lda = pickle.load(open("model/lda_model.pkl", "rb"))
    topic_dist = list(lda.transform(new_vec)[0])
    store_vals = list()

    for i in range(len(covid_papers)):
        if(i in covid_papers.index):
            store_vals.append((int(i), (sqrt(mean_squared_error(topic_dist, [float(i) for i in covid_papers.loc[int(i), 'topic_dist'].strip('[]').split(', ')])))))
    most_similar = sorted(store_vals, key=itemgetter(1))

    if n_clicks is None:
        data = [covid_papers.loc[int(i[0]), ['title', 'url', 'abstract', 'authors' ,'publish_time']] for i in most_similar[1:(n_articles+1)]]
    else:
        data = [covid_papers.loc[int(i[0]), ['title', 'url', 'abstract', 'authors' ,'publish_time']] for i in most_similar[1:(n_articles+1 +(n_articles*n_clicks))]]
    if load_page is None:
        raise PreventUpdate
    else:
        out = []
        for info in data:
            out.append(dbc.ListGroup([
                dbc.ListGroupItem(
                    [
                        dbc.ListGroupItem(dbc.ListGroupItemHeading(info[0]), href=info[1], target="_blank"),
                        html.Br(),
                        dbc.ListGroupItemText(f"Authors: {info[3]}", style = {'text-align': 'left', 'font-size': 'small'}),
                        dbc.ListGroupItemText(f"Published: {info[4]}", style={'text-align': 'left', 'font-size': 'smaller'}),
                        dbc.ListGroupItemText(info[2], style={'text-align':'justify'}),
                    ])], flush=True))
    out.append(html.Br())
    out.append(dbc.Button("Load More", id='button', outline=True, color="primary", className="mr-1", style={'margin-left': '46%',
                                                                                                    'margin-bottom': '10%px',
                                                                                                    'verticalAlign': 'middle'}))
    return out



if __name__ == '__main__':
    app.run_server(debug=True)
