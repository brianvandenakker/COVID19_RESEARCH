import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt


app = dash.Dash(suppress_callback_exceptions=True)

covid_papers = pd.read_csv("data/covid_papers.csv", index_col=[0])

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.H1('Find Your Next Read', style = {'text-align': 'center', 'font-family':'sans-serif'}),
    html.H4("Search the COVID-19 Research Database or Enter the Text From a Paper You're Interested In. We'll Read the Paper You're Interested in, and Provide Recommendations for Further Reading Based on Those Interests",
            style = {'text-align': 'center', 'font-family':'sans-serif'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Search Database', value='tab-1'),
        dcc.Tab(label='Enter Your Own Paper', value='tab-2'),
    ], style = {'width':'50%', 'margin':'auto'}),
    html.Br(),
    html.Div(id='tab-output'),
])

@app.callback(Output('tab-output', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
                html.Div([dcc.Dropdown(id="paper-search",
                options = [{'label': f"{title}", 'value':f"{index}"} for index, title in enumerate(covid_papers['title'])],
                placeholder = "Search Papers")], style = {'width':'50%' , 'margin':'auto'}),
                html.Div(html.Ul(id = 'similar-papers-db'), style={'font-family':'sans-serif', 'width': '70%', 'margin':'10%'}),
                html.Button('Load More', id='button', style={'margin':'auto'})])
    elif tab == 'tab-2':
        return html.Div([dcc.Textarea(
                id='text-box',
                placeholder = "Enter The Text You'd Like us to Read Here. We'll Process the Text and Return the Papers from the COVID-19 Database which are Closest to Your Entry. ",
                style={'width': '100%', 'height': 300}),
                html.Div(html.Ul(id='similar-papers-new'), style={'text-align':'center'})
                ])
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
        out.append(html.Div([
                    html.Li([
                        html.A(html.H2(info[0]), href=info[1], target="_blank"),
                    ]),
                    html.Li([
                        html.P(info[3]),
                        html.P(f"Published: {info[4]}", style={'text-align': 'left', 'font-size': 'smaller'})
                    ]),
                    html.Li([
                        html.P(info[2], style={'text-align': 'justify'})
                    ]),
                    html.Br()
                ]))
    return out

@app.callback(
    Output(component_id='similar-papers-new', component_property= 'children'),
    [Input(component_id='text-box', component_property='value')]
)
def similar_text_new(process_text, n_articles = 10):
    if process_text is None:
        raise PreventUpdate

    new_vec = vec.transform(text_to_compare)
    topic_dist = list(lda.transform(new_vec)[0])
    store_vals = list()
    topic_dist = [float(i) for i in covid_papers.loc[int(paper_search), 'topic_dist'].strip('[]').split(', ')]

    for i in range(len(covid_papers)):
        if(i in covid_papers.index):
            store_vals.append((int(i), (sqrt(mean_squared_error(topic_dist, [float(i) for i in covid_papers.loc[int(i), 'topic_dist'].strip('[]').split(', ')])))))
    most_similar = sorted(store_vals, key=itemgetter(1))

    data = [covid_papers.loc[int(i[0]), ['title', 'url', 'abstract', 'authors' ,'publish_time']] for i in most_similar[1:n_articles+1]]
    out = []
    for info in data:
        out.append(html.Div([
                    html.Li([
                        html.A(html.H2(info[0]), href=info[1], target="_blank"),
                    ]),
                    html.Li([
                        html.P(info[3]),
                        html.P(f"Published on: {info[4]}", style={'text-align': 'right'})
                    ]),
                    html.Li([
                        html.P(info[2], style={'text-align': 'justify'})
                    ])
                ]))
    return out



if __name__ == '__main__':
    app.run_server(debug=True)
