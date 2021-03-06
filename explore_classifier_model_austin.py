'''
Interactive dashboard to show characteristics of a classification model
for emails. Will display what emails were successfully categorized and which
weren't and which terms are most important for classifier.
'''

#----------------------------------------------------------------------#
# Libraries and dependencies                                           #
#----------------------------------------------------------------------#
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import sys
import os
from flask import send_from_directory
import pickle
import nltk
from nltk.corpus import stopwords
import getopt
#
# def usage():
#     sys.stdout.write("Usage: python viz_classifyer.py [-d|--directory= <top directory of the github repository where your directory yaml sits>] [-n|--number= <number of output emails requested>] [-h|?|--help]")
#     return True

#---------------------------------------------------#
# Return name of dataframe based on input parameter #
#---------------------------------------------------#

def chooseDF(responseClass):

    global resultsDF_tp
    global resultsDF_fp
    global resultsDF_tn
    global resultsDF_fn

    print(type(resultsDF_tn))
    if responseClass == 'truePositive':
        return (resultsDF_tp)
    elif responseClass == 'falsePositive':
        return (resultsDF_fp)
    elif responseClass == 'falseNegative':
        return (resultsDF_fn)
    elif responseClass == 'trueNegative':
        return (resultsDF_tn)
    else:
        return Null



#------------------------------------------------------#
# Prepare input data                                   #
#Load results from classifier notebook
#------------------------------------------------------#

stopwords_english = stopwords.words('english')

# try:
#     opts, args = getopt.getopt(sys.argv[1:], "d:n:h?", ["--directory=", "--number=", "--help"])
# except getopt.GetoptError as err:
#     #Exit if can't parse args
#     usage()
#     sys.exit(2)
# for o, a in opts:
#     if (o == '-h' or o == '-?'):
#         usage()
#         exit(0)
#     elif o in ('-d', '--directory'):
#         parent_path = a
#         sys.path.insert(0, parent_path + '//' + 'utils')
#         from load_directories import directory_loader
#         input_directory, output_directory = directory_loader(parent_path)
from utils.viz_utils import highlightTerms, formatEmail
from utils.viz_utils import countTokens, countFreqs
from utils.viz_utils import generateAccuracyTable,generateTruthTable
from utils.viz_utils import load_term_scores
from utils.viz_utils import generateTermsTable

output_directory='/Users/austinlasseter/github_repos/federal_register'
# #Load term scores from csv
termScores=pd.read_csv(output_directory + '/' + 'termScores.csv', index_col=None)

#Load dict of accuracy scores for classifiery
with open(output_directory + '/' + 'classifierStats.pyc', 'rb') as f:
    classifierStats = pickle.load(f)
    f.close()

#Load results of testing each test doc
with open(output_directory + '/' + 'classifierTestResults.pyc', 'rb') as f:
    classifierTestResults = pickle.load(f)
    f.close()

#-------------------------------------------------------------------------#
# Make 4 chunks of the test data, depending on truth value                #
#-------------------------------------------------------------------------#
resultsDF_tp = classifierTestResults[classifierTestResults['truthValue'] == 'truePositive']
resultsDF_fp = classifierTestResults[classifierTestResults['truthValue'] == 'falsePositive']
resultsDF_tn = classifierTestResults[classifierTestResults['truthValue'] == 'trueNegative']
resultsDF_fn = classifierTestResults[classifierTestResults['truthValue'] == 'falseNegative']


responseTypes = ['truePositive', 'trueNegative', 'falsePositive', 'falseNegative']

#-------------------------------------------------------------------#
# Store all terms in list for easy access                           #
#-------------------------------------------------------------------#
termsList = termScores['term'].tolist()
selectedDF = resultsDF_tp
# numEmailsInSelectedDF = selectedDF.shape[0]
numEmailsInSelectedDF = len(selectedDF)
emailPointer = 1

#Highlight the terms in the email which are in the visible list
# highlightedEmailSubject = highlightTerms(resultsDF_tp.iloc[(emailPointer - 1)].subject, termsList, stopwords_english)
highlightedEmailBody = highlightTerms(resultsDF_tp.iloc[(emailPointer - 1)].abstract, termsList, stopwords_english)
posScore = selectedDF.iloc[(emailPointer - 1)].posProbability
negScore = selectedDF.iloc[(emailPointer - 1)].negProbability
# subjectPlusBody = (resultsDF_tp.iloc[(emailPointer -1)].abstract)

#------------------------------------------------------------------------#
#Local version of stylesheet https://codepen.io/chriddyp/pen/bWLwgP.css #
#------------------------------------------------------------------------#
stylesheets = ['bWLwgP.css']

#--------------------------------------------#
# Start building the dashboard - initialize  #
#--------------------------------------------#
app = dash.Dash()
app.title = "Explore Email Classifier Performance"

#--------------------------------------------#
#Allow locally-served css
#--------------------------------------------#
app.css.config.serve_locally=True
app.scripts.config.serve_locally=True

@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

#-----------------------------------------------------#
# Layout dashboard with HTML and dash core components #
#-----------------------------------------------------#
app.layout = html.Div(children = [
    html.Link(
        rel='stylesheet',
        href='/static/bWLwgP.css'
    ),
    html.Link(
        href='/static/sorttable.js'
    ),
    html.Div(id="bannerDiv",
             children = [
                 html.Img(id = "contextEdgeLogo",
                          src='./static/ContextEdge.png'),
                 html.H1(id = "appHeader",
                         children = "How Well is My Email Classifier Working?",
                         style={'textAlign': 'center'}) #H1
             ]), #bannerDiv,
    html.H2("Overall Classifier Performance"),
    html.Div(id="classifier_stats_div", children =
             [
                 html.Div(id="performanceDiv",
                          children = [
                              html.Div(
                                  id="accuracy_table_div",
                                  children = [
                                      generateAccuracyTable(classifierStats)
                                  ],
                                  className = "six columns"
                              ),
                              html.Div(
                                  id = "truth_table_div",
                                  children = generateTruthTable(classifierStats),
                                  className = "six columns"
                              )
                          ],
                          className="row")
             ]), #classifier_stats_div
    html.Div(id="text_and_graph_div", children=[
        html.Div(id="holder_div", className = "six columns", children = [
            html.H2("Performance On Each Email"),
            html.Div(id="output_class_selector_div", children=[
                html.Label("Show me..."),
                dcc.RadioItems(
                    id="showMe",
                    options = [{'label': "{}s".format(responseType), 'value': responseType } for responseType in responseTypes ],
                    value = 'truePositive',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Table(id = 'tableJumpTo', children = [
                html.Tr(children = [
                    html.Th(html.Label("Jump to Email No.")),
                    html.Th(dcc.Input(id='inputEmailNumber', value = 1, type='number')), #Returns unicode string even though we request a number!
                    html.Th(html.P(id = 'pTotalEmails', children = " of {}".format(numEmailsInSelectedDF))),
                    html.Th(html.Button(id='buttonSubmit', children="Submit")),
                ]) #tr
            ],
            style={'margin-left': '0px'}), #tableJumpTo
            html.Div(id="text_div", children=[
                html.Iframe(
                    id = 'email_text_iframe',
                    sandbox='',
                    srcDoc = formatEmail(posScore,
                                         negScore,
                                         # highlightedEmailSubject,
                                         highlightedEmailBody),
                    style = {'width': '650px', 'height': '200px'}
                )
            ], style= {'height':'200px', 'padding-top': '20px'})
        ]), #holder div
    ]),
    html.H2("Terms/features used in model:"),
    html.Div
    (
        id="tableAndSortdiv",
        className = "six columns",
        children=
        [
            html.Div
            (
                id = "sortSelectDiv",
                style = {'display': 'block'},
                children =
                [
                    html.Label("Sort By..."),
                    html.Div
                    (
                        id="sortOptionsDiv",
                        children =
                        [
                            dcc.RadioItems
                            (
                                id="sortBy",
                                style={'display': 'inline-block', 'float': 'left'},
                                options = [{'label': "{}".format(col), 'value': col } for col in termScores.columns ],
                                value = 'modelCoef',
                                labelStyle={'display': 'inline-block'}
                            ),
                            dcc.RadioItems
                            (
                                id = "sortOrder",
                                style = {'display': 'inline-block', 'float': 'left'},
                                options = [{'label': "{}".format(col), 'value': col } for col in ['Ascending', 'Descending']],
                                value = 'Ascending',
                                labelStyle={'display': 'inline-block'}
                            )
                        ]
                    ), #SortoptionsDiv
                    html.Div
                    (
                        id = "table_div",
                        style= { 'clear': 'left', 'overflow-y': 'scroll', 'height': '350px'},
                        children = [generateTermsTable(termScores)]
                    ) #table_div
                ]

            ), #sortSelectDiv
        ]
    ) #tableAndSortDiv
])

#-------------------------------------------------------------------#
# Define interactive behaviors from inputs                          #
#-------------------------------------------------------------------#

#-------------------------------------------------------------------#
# callbacks for radio button to select email subset and number      #
#-------------------------------------------------------------------#
@app.callback(Output('pTotalEmails', 'children'),
              [Input('showMe', 'value')])
def update_df_selection(input1):
    global resultsDF_tp
    global resultsDF_tn
    global resultsDF_fn
    global resultsDF_fp
    global emailPointer

    selectedDF = chooseDF(input1)#

    #Reset to ist email
    emailPointer = 1

    # return (" of {}".format(selectedDF.shape[0]))
    return (" of {}".format(len(selectedDF)))

#------------------------------------------------------------#
# Update the text in the iframe                              #
# depending on which class of data and number email selected #
#------------------------------------------------------------#
@app.callback(Output('email_text_iframe', 'srcDoc'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value'),
               State('inputEmailNumber', 'value')])
def update_displayed_email_text(nClicks, inputDF, inputEmailNumber):
    global resultsDF_tp
    global resultsDF_tn
    global resultsDF_fn
    global resultsDF_fp
    global selectedDF
    global emailPointer
    global termsList

    #Switch to selected type of emails, true positive, false pos, etc
    selectedDF = chooseDF(inputDF)

    # if (int(inputEmailNumber) > selectedDF.shape[0]):
    if (int(inputEmailNumber) > len(selectedDF)):
        emailPointer = 1
    else:
        emailPointer = int(inputEmailNumber)

    posProbability = selectedDF.iloc[(emailPointer - 1)].posProbability

    negProbability = selectedDF.iloc[(emailPointer - 1)].negProbability

    # highlightedEmailSubject = highlightTerms(selectedDF.iloc[(emailPointer - 1)].subject, termsList, stopwords_english)
    highlightedEmailBody = highlightTerms(selectedDF.iloc[(emailPointer - 1)].abstract, termsList, stopwords_english)

    return(formatEmail(posProbability,
                       negProbability,
                       # highlightedEmailSubject,
                       highlightedEmailBody))


#---------------------------------------------------------------#
# Highlight accuracy or error rate depending on which value     #
# chosen in radio button                                        #
# Feeling bad about writing four functions to update four cells #
# but can't see better way...                                   #
#---------------------------------------------------------------#
@app.callback(Output('accuracyTableCell2', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateAccuracyCell2(n_clicks, showMe):
    if (showMe in ['truePositive', 'trueNegative']):
        return('highlightedCell')
    else:
        return('normalCell')
@app.callback(Output('errorTableCell2', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateErrorCell2(n_clicks, showMe):
    if (showMe in ['truePositive', 'trueNegative']):
        return('normalCell')
    else:
        return('highlightedCell')

#---------------------------------------------------------------#
# Highlight truth table depending on radio button selection     #
#---------------------------------------------------------------#
@app.callback(Output('truePositivesCell', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateTruePositivesCell(n_clicks, showMe):
    if (showMe == 'truePositive'):
        return('highlightedCell')
    else:
        return('normalCell')
@app.callback(Output('trueNegativesCell', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateTrueNegativesCell(n_clicks, showMe):
    if (showMe == 'trueNegative'):
        return('highlightedCell')
    else:
        return('normalCell')
@app.callback(Output('falsePositivesCell', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateFalsePositivesCell(n_clicks, showMe):
    if (showMe == 'falsePositive'):
        return('highlightedCell')
    else:
        return('normalCell')
@app.callback(Output('falseNegativesCell', 'className'),
              [Input('buttonSubmit', 'n_clicks')],
              [State('showMe', 'value')])
def updateFalseNegativesCell(n_clicks, showMe):
    if (showMe == 'falseNegative'):
        return('highlightedCell')
    else:
        return('normalCell')

#------------------------------------------------------------------------#
# Sort and return the terms table using the options in the radio buttons #
#------------------------------------------------------------------------#
@app.callback(Output('table_div', 'children'),
              [Input('sortBy', 'value'),
              Input('sortOrder', 'value')])
def sortTermsTable(mySortBy, mySortOrder):
    print("{}|{}".format(mySortBy, mySortOrder))
    return(generateTermsTable(termScores, mySortBy, mySortOrder))

app.run_server(debug=True)
