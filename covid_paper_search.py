import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.metrics import mean_squared_error
from math import sqrt


app = dash.Dash()

covid_papers = pd.read_csv("data/metadata.csv")

app.layout = html.Div([
    html.Div([dcc.Dropdown(id="paper_search",
        options = [{'label': f"{title}", 'value':f"{index}"} for index, title in enumerate(covid_papers['title'])],
                   placeholder = "Search Papers")
                   ], style = {'width':'50%' , 'argin':'auto'}),
    #html.Div(dcc.Input(id='paper_search', type='text', placeholder = "Search Papers"), style={'text-align':'center', 'font-family':'sans-serif'}),
    #html.Button('Search', id='button', style={'margin':'auto', 'font-family':'sans-serif'}),
    html.Div([
        dcc.Markdown(id="similar_papers")
        ], style={'text-align':'center', 'font-family':'sans-serif'})
])


@app.callback(
    Output(component_id='similar_papers', component_property= 'children'),
    #[Input(component_id='button', component_property='n_clicks')],
    [Input(component_id='paper_search', component_property='value')]
)

def similar_text(paper_search, n_articles = 10):
    store_vals = list()
    if(isinstance(paper_search, list) and len(paper_search) is 1):
        pass
        #new_vec = vec.transform(paper_search)
        #topic_dist = list(lda.transform(new_vec)[0])
    #elif(isinstance(paper_search, int)):
    #    topic_dist = covid_papers.loc[paper_search, 'topic_dist']
    else:
        #topic_dist = covid_papers.loc[paper_search, 'topic_dist']
        #raise ValueError
        pass
    topic_dist = covid_papers.loc[paper_search, 'topic_dist']
    for i in range(len(covid_papers)):
        if(i in covid_papers.index):
            store_vals.append((i, (sqrt(mean_squared_error(topic_dist, covid_papers.loc[i, 'topic_dist'])))))
    most_similar = sorted(store_vals, key=itemgetter(1))
    return [covid_papers.loc[i[0]] for i in most_similar[1:n_articles+1]]


if __name__ == '__main__':
    app.run_server(debug=True)
