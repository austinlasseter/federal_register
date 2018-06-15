import dash
import dash_core_components as dcc
import dash_html_components as html
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

#Load the pickled files
with open('classifierTestResults.pyc', 'rb') as f:
    classifierTestResults = pickle.load(f)
    f.close()
with open('top10.pyc', 'rb') as f:
    top10 = pickle.load(f)
    f.close()
y_test=classifierTestResults['ground_truth']
predictions=classifierTestResults['predicted_value']
probabilities=classifierTestResults['posProbability']
# ROC-AUC Score

FPR, TPR, _ = roc_curve(y_test, probabilities)
roc_score=round(100*roc_auc_score(y_test, predictions),1)


import plotly
import plotly.graph_objs as go

app = dash.Dash()
application = app.server


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


app.layout = html.Div(children=[
    html.H1(children='Model Performance'),

    html.Div(children='''
        NLP Classification of Federal Register Abstracts.
    '''),
    html.Div(children=[
    dcc.Graph(
        id='roc-auc',
        figure={
            'data': [{
                'x': FPR,
                'y': TPR,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'AUC: '+str(roc_score)
            },
            {
                'x': [0,1],
                'y': [0,1],
                'type': 'scatter',
                'mode': 'lines', # 'markers' | 'lines+markers'
                'name': 'Baseline Area: 50.0'
            }],
            'layout': {
                'title': 'Receiver Operating Characteristic (ROC) Curve',
                'xaxis': {
                    'title': 'False Positive Rate (100-Specificity)',
                    'scaleratio': 1,
                    'scaleanchor': 'y'
                },
                    'yaxis': {
                        'title': 'True Positive Rate (Sensitivity)'
                    }
                }
            }, className="six columns"
        ),
        dcc.Graph(
            id='feature-importance',
            figure={
                'data': [{
                    'x': top10['term'],
                    'y': top10['modelCoef'],
                    'type': 'bar'
                }],
                'layout': {
                    'title': 'Most Important Features',
                    'xaxis': {
                        'title': 'Feature',
                        'scaleratio': 1,
                        'scaleanchor': 'y'
                    },
                        'yaxis': {
                            'title': 'Logistic Regression Model Coefficient'
                        }
                    }
                }, className="six columns"
            ),
    ], className='Row'
    )

])


if __name__ == '__main__':
    application.run(debug=True, port=8050)
