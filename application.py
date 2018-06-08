
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


import json
import requests
from bs4 import BeautifulSoup
import re
import urllib
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import base64
# import matplotlib
# matplotlib.use('qt4agg')
# import matplotlib.pyplot as plt

#----------------------------------------------------------------------#
# Get Federal Data                                                     #
#----------------------------------------------------------------------#

# Pull from fedreg API: all agencies
number=2000
page=1
url='https://www.federalregister.gov/api/v1/documents.json?order=relevance&page='+str(page)+'&per_page='+str(number)
# get request
response = requests.get(url).json()
# Flatten the dataset, WITHOUT creating duplicate rows when two agencies are listed.
df = json_normalize(response['results'])
# Flatten the "agencies" dictionary column into a new dataset
df2 = pd.DataFrame(df['agencies'].values.tolist(), index=df.index)
# Extract the name of the sub-agency as a separate column
def extractor (col):
    try:
        return(col['name'])
    except:
        pass
df2['subagency']=df2[1].apply(extractor)
# Flatten the first agency and its metadata into a separate dataset
df3=pd.DataFrame(df2[0].values.tolist(), index=df2.index)
# Join the first-agency dataset with the subagency column
df4=df3.join(df2, how='outer').drop([0,1], axis=1)
# Join that dataset into the original dataset of Registry entries
df5=df.join(df4,how='outer').drop('agencies', axis=1)
# Fix the column names
df5.rename(columns={'name':'agency'}, inplace=True)
df5.rename(columns={'id':'agency_id'}, inplace=True)
# large dataset
# df5=df5[['document_number', 'publication_date', 'agency_id', 'agency', 'subagency',
#         'title', 'abstract', 'type', 'excerpts', 'raw_name', 'slug', 'url', 'html_url',
#         'pdf_url', 'public_inspection_pdf_url',  'json_url', 'parent_id']]
# small dataset
df_a=df5[['abstract', 'type', 'agency']]

############## Pull from fedreg API: all agencies
number=2000
page=2
url='https://www.federalregister.gov/api/v1/documents.json?order=relevance&page='+str(page)+'&per_page='+str(number)
# get request
response = requests.get(url).json()
# Flatten the dataset, WITHOUT creating duplicate rows when two agencies are listed.
df = json_normalize(response['results'])
# Flatten the "agencies" dictionary column into a new dataset
df2 = pd.DataFrame(df['agencies'].values.tolist(), index=df.index)
# Extract the name of the sub-agency as a separate column
def extractor (col):
    try:
        return(col['name'])
    except:
        pass
df2['subagency']=df2[1].apply(extractor)
# Flatten the first agency and its metadata into a separate dataset
df3=pd.DataFrame(df2[0].values.tolist(), index=df2.index)
# Join the first-agency dataset with the subagency column
df4=df3.join(df2, how='outer').drop([0,1], axis=1)
# Join that dataset into the original dataset of Registry entries
df5=df.join(df4,how='outer').drop('agencies', axis=1)
# Fix the column names
df5.rename(columns={'name':'agency'}, inplace=True)
df5.rename(columns={'id':'agency_id'}, inplace=True)
# large dataset
# df5=df5[['document_number', 'publication_date', 'agency_id', 'agency', 'subagency',
#         'title', 'abstract', 'type', 'excerpts', 'raw_name', 'slug', 'url', 'html_url',
#         'pdf_url', 'public_inspection_pdf_url',  'json_url', 'parent_id']]
# small dataset
df_b=df5[['abstract', 'type', 'agency']]


############## Pull from fedreg API: all agencies
number=2000
page=3
url='https://www.federalregister.gov/api/v1/documents.json?order=relevance&page='+str(page)+'&per_page='+str(number)
# get request
response = requests.get(url).json()
# Flatten the dataset, WITHOUT creating duplicate rows when two agencies are listed.
df = json_normalize(response['results'])
# Flatten the "agencies" dictionary column into a new dataset
df2 = pd.DataFrame(df['agencies'].values.tolist(), index=df.index)
# Extract the name of the sub-agency as a separate column
def extractor (col):
    try:
        return(col['name'])
    except:
        pass
df2['subagency']=df2[1].apply(extractor)
# Flatten the first agency and its metadata into a separate dataset
df3=pd.DataFrame(df2[0].values.tolist(), index=df2.index)
# Join the first-agency dataset with the subagency column
df4=df3.join(df2, how='outer').drop([0,1], axis=1)
# Join that dataset into the original dataset of Registry entries
df5=df.join(df4,how='outer').drop('agencies', axis=1)
# Fix the column names
df5.rename(columns={'name':'agency'}, inplace=True)
df5.rename(columns={'id':'agency_id'}, inplace=True)
# large dataset
# df5=df5[['document_number', 'publication_date', 'agency_id', 'agency', 'subagency',
#         'title', 'abstract', 'type', 'excerpts', 'raw_name', 'slug', 'url', 'html_url',
#         'pdf_url', 'public_inspection_pdf_url',  'json_url', 'parent_id']]
# small dataset
df_c=df5[['abstract', 'type', 'agency']]

# Put them all together
df=pd.concat([df_a, df_b, df_c], ignore_index=True)



#----------------------------------------------------------------------#
# Prep, Split, and Vectorize                                           #
#----------------------------------------------------------------------#

# Remove rows with empty data
df=df.dropna(subset=['abstract'], how='all')
# Create the target variable
df['target']=False
list_of_agencies=['Transportation Department',
                'Commerce Department',
                'Health and Human Services Department',
                'Homeland Security Department',
                'Energy Department']
for dept in list_of_agencies:
    df.loc[df['agency']==dept, 'target']=True
print(df['target'].value_counts())

# Add to stopwords (this should become user-input values)
new_stopwords=['and', 'of', 'or', 'the']
new_stopwords += stopwords.words('english')
# Preprocess
from utils.email_utils import stripPunctuation
df['abstract'] = df['abstract'].apply(stripPunctuation)
from utils.email_utils import preprocess
df['abstract'] = df['abstract'].apply(preprocess)


tvec=TfidfVectorizer(strip_accents='unicode', lowercase=True, preprocessor=None, analyzer='word',
                    tokenizer=None, stop_words='english',  max_df=0.8, max_features=None)

# Split & vectorize
X=df['abstract']
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

tvec_train=tvec.fit_transform(X_train)
feature_names = tvec.get_feature_names()
tvec_test=tvec.transform(X_test)

#----------------------------------------------------------------------#
# Logistic Regression Classifier                                       #
#----------------------------------------------------------------------#
# Set up the parameters for the gridsearch.
gs_params = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,0,100)
}
# Instantiate the gridsearch
lr_gridsearch = GridSearchCV(LogisticRegression(), param_grid=gs_params, cv=5, verbose=1)
# Fit the instantiated gridsearch to the vectorized training data
lr_gridsearch.fit(tvec_train, y_train)
# Instantiate model using those parameters
model_lr = lr_gridsearch.best_estimator_
# Predict on the test data
predictions=model_lr.predict(tvec_test)
# Probabilities
probabilities = model_lr.predict_proba(tvec_test)[:,1]
# Convert each component to a pandas dateframe
df_probs=pd.DataFrame(probabilities, columns=['probabilities']).reset_index(drop=True)
df_preds=pd.DataFrame(predictions, columns=['predictions']).reset_index(drop=True)
df_Xtest=pd.DataFrame(X_test).reset_index(drop=True)
df_ytest=pd.DataFrame(y_test).reset_index(drop=True)
# metrics

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# Reset a new index because we removed all the training data but never reset the index, so it has gaps
# and drop=True gets rid of the old index
final=pd.concat([df_Xtest, df_ytest, df_probs, df_preds], axis=1)
#Now let's see how well this model did, true positives, false positives, etc
def accuracy(tp, tn, fp, fn):
    return ((tp + tn)/(tp + tn + fp + fn))

def error_rate(tp, tn, fp, fn):
    return ((fp + fn)/ (tp + tn + fp + fn))

true_positives = len(final.loc[(final['target'] == True) & (final['predictions'] == True)])
false_positives = len(final.loc[(final['target'] == False) & (final['predictions'] == True)])
true_negatives = len(final.loc[(final['target'] == False) & (final['predictions'] == False)])
false_negatives = len(final.loc[(final['target'] == True) & (final['predictions'] == False)])

#First get the predicted class label for each document
predictions = model_lr.predict(tvec_test)
print("Accuracy score for your classifier: {:.3f}\n".format(model_lr.score(tvec_test, y_test)))
print("Error rate for your classifier: {:.3f}\n".format(1-model_lr.score(tvec_test, y_test)))
classifierStats = dict()
classifierStats['accuracy'] = model_lr.score(tvec_test, y_test)
classifierStats['errorRate'] = (1 - model_lr.score(tvec_test, y_test))

 #Also store the predicted class probabilities
predictProbabilities = model_lr.predict_proba(tvec_test)
# Function for Truth Value
def truth_value(myRow):
    if (myRow['ground_truth'] == True and myRow['predicted_value'] == True):
        return 'truePositive'
    elif (myRow['ground_truth'] == True and myRow['predicted_value'] == False):
        return 'falseNegative'
    elif (myRow['ground_truth'] == False and myRow['predicted_value'] == True):
        return 'falsePositive'
    elif (myRow['ground_truth'] == False and myRow['predicted_value'] == False):
        return 'trueNegative'
    else:
        return None


# Define the results dataset
y_test_array=y_test.values
results = [(y_test_array[i], predictions[i]) for i in range(0,len(predictions))]
# add in from records
enrichedResults = pd.DataFrame.from_records(results,
    columns = ['ground_truth', 'predicted_value'])
enrichedResults['truthValue'] = enrichedResults.apply(lambda row: truth_value(row), axis=1)
enrichedResults['abstract'] = X_test.tolist()
enrichedResults['posProbability'] = [prob[1] for prob in predictProbabilities]
enrichedResults['negProbability'] = [prob[0] for prob in predictProbabilities]
counts = enrichedResults['truthValue'].value_counts()
for i in range(0,len(counts)):
    classifierStats[counts.index[i]] = counts[i]


#---------------------------------------------------#
# Evaluation and Feature importance                 #
#---------------------------------------------------#


# Create a dataframe of the features by importance
importance=pd.DataFrame(sorted(zip(model_lr.coef_[0], feature_names)), columns=['modelCoef', 'term'])
# 10 Features most associated with target=False
high5=pd.DataFrame(importance.head(5))
low5=pd.DataFrame(importance.tail(5))
top10=pd.concat([high5, low5], ignore_index=True)

# 100 Features most associated with target=False
high50=pd.DataFrame(importance.head(50))
low50=pd.DataFrame(importance.tail(50))
termScores=pd.concat([high50, low50], ignore_index=True)

# Feature importance graphic
#
# import seaborn as sns
# sns.set(style="whitegrid", color_codes=True)
#
# ax = top10.set_index('term').plot(kind='bar', legend=False, fontsize=18, figsize=(15, 7))
# plt.title('Features with greatest predictive power',  fontsize=19)
#
# plt.xticks(rotation = 45,  fontsize=18)
# plt.xlabel('Features least or most associated with target', fontsize=18)
# plt.yticks(rotation = 0,  fontsize=18)
# plt.ylabel('Coefficient', rotation=90,  fontsize=18)
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
feature_import = base64.b64encode(open('feature_importance.png', 'rb').read())

# ROC-AUC Score
FPR = dict()
TPR = dict()
ROC_AUC = dict()
# For the target class, find the area under the curve:
FPR[1], TPR[1], _ = roc_curve(y_test, probabilities)
ROC_AUC[1] = auc(FPR[1], TPR[1])
# # Let's draw that:
# plt.style.use('seaborn-white')
# plt.figure(figsize=[11,9])
# plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
# plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=18)
# plt.ylabel('True Positive Rate', fontsize=18)
# plt.title('ROC Curve', fontsize=18)
# plt.legend(loc="lower right", fontsize=18);
# plt.savefig('rocauc.png', dpi=300, bbox_inches='tight')
rocauc_image = base64.b64encode(open('rocauc.png', 'rb').read())


#---------------------------------------------------#
# Pickling and unpickling (this is a temp workaround) #
#---------------------------------------------------#

with open('classifierStats.pyc', 'wb') as f:
    pickle.dump(classifierStats, f)
f.close()
with open('classifierTestResults.pyc', 'wb') as f1:
    pickle.dump(enrichedResults, f1)
f1.close()
# These pickled files get called further down. This is a temp workaround/legacy.



#######################################################
#------------------------------------------------------#
# Dashboard App                                        #
#------------------------------------------------------#
print('this is where the fun begins')

# logo
deloitte = base64.b64encode(open('Deloitte_Logo.png', 'rb').read())
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

from utils.viz_utils import highlightTerms, formatEmail
from utils.viz_utils import countTokens, countFreqs
from utils.viz_utils import generateAccuracyTable,generateTruthTable
from utils.viz_utils import load_term_scores
from utils.viz_utils import generateTermsTable

#Load the pickled files (this needs to get updated and replaced)
with open('classifierStats.pyc', 'rb') as f:
    classifierStats = pickle.load(f)
    f.close()

with open('classifierTestResults.pyc', 'rb') as f:
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

highlightedEmailBody = highlightTerms(resultsDF_tp.iloc[(emailPointer - 1)].abstract, termsList, stopwords_english)
posScore = selectedDF.iloc[(emailPointer - 1)].posProbability
negScore = selectedDF.iloc[(emailPointer - 1)].negProbability

#------------------------------------------------------------------------#
#Local version of stylesheet https://codepen.io/chriddyp/pen/bWLwgP.css #
#------------------------------------------------------------------------#
stylesheets = ['bWLwgP.css']

#--------------------------------------------#
# Start building the dashboard - initialize  #
#--------------------------------------------#
app = dash.Dash(__name__)
application=app.server
app.title = "ContextEdge Rocks!"

#--------------------------------------------#
#Allow locally-served css
#--------------------------------------------#
# app.css.config.serve_locally=True
# app.scripts.config.serve_locally=True
#
# @app.server.route('/static/<path:path>')
# def static_file(path):
#     static_folder = os.path.join(os.getcwd(), 'static')
#     return send_from_directory(static_folder, path)

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
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(deloitte.decode()), style={'width': '300px'})
        ]),
    html.Div(id="bannerDiv",
             children = [
                 html.H1(id = "appHeader",
                         children = "Classifying Federal Register Abstracts",
                         style={'textAlign': 'center'}) #H1
             ]), #bannerDiv,
    html.Div("This app pulls 6,000 abstracts from the Federal Register, labeled \
    according to the publishing agency. The app trains a logistic regression \
    classifier on 70 percent of the data, then predicts the other 30 percent.\
    The top 100 most important features are listed below and highlighted within \
    each abstract; feature with zero importance are highlighted in blue."),
    html.Div("Source of Text Data:"),
    html.A('Federal Register API', href='https://www.federalregister.gov/developers/api/v1'),
    html.Div("Target Class includes abstracts from top 5 agencies (about half the dataset):"),
    html.Div(list_of_agencies),
    html.Br(),
    html.H2("Overall Classifier Performance"),
    html.Br(),
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
            html.H2("Performance On Each Abstract"),
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
                    html.Th(html.Label("Jump to Abstract No.")),
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
                ),

            ], style= {'height':'200px', 'padding-top': '20px'})
        ]), #holder div
    ]),
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(rocauc_image.decode()), style={'width': '400px'}),
        html.Img(src='data:image/png;base64,{}'.format(feature_import.decode()), style={'width': '600px'})
    ], className="Row"),
    html.H2("Features used in model:"),
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

if __name__=='__main__':
    application.run(debug=True, port=8080)
