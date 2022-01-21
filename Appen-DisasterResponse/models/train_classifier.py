# Project: Building a Disaster Response Data Pipeline

"""
A python script that shows the Machine Learning Pipeline processes:

- it takes the database file path and model file path, 
- creates and trains a classifier, 
- and stores the classifier into a pickle file to the specified model file path


Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response.db ML_classifier.pkl

INPUTS:
    1) Path to SQLite destination database (e.g. disaster_response.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. ML_classifier.pkl)
"""


#=============================================================================
#import libraries for loading and manipulating data
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

#import libraries for text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#import libraries for our Machine Learning model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report
import pickle

import warnings
import sys
import os
import re

warnings.simplefilter('ignore')

#=============================================================================

def load_data(database_filepath):
    """
    Function that loads data from the SQLite Database into a pandas dataframe
    
    INPUT:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response.db)
    OUTPUT:
        X -> a dataframe containing the independent variable
        y -> a dataframe containing the multiple dependent variables
        category_names -> List of strings (categories name)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('message_disaster_response', engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names



def tokenize(text):
    """
    Function to tokenize the text (message) function
    
    INPUT:
        text -> Text messages which needs to be tokenized
    OUTPUT:
        clean_tokens -> List of tokens extracted from the provided text
    """
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    #search for urls in message and replace with urlplaceolder
    #url_rgex is the regular expression to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    #Normalization & capitalization:
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize text: split into words
    words = word_tokenize(text)
  
    #Remove stop words
    words = [w for w in words if w not in stop_words]
     
    # lemmatize as shown in course note
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    

    return clean_tokens
    


def build_model():
    """
    Funtion to build the machine learning pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # best parameter estimated from grid search
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model,X_true,y_true):
    """
    Function to generate classification report on a model
    
    INPUT:
        Model, true data sets, that is X_test & y_test data sets
    OUTPUT: 
        Prints the f1 score, precision and recall for each output category
    """
    y_pred = model.predict(X_true)
    for i, col in enumerate(y_true):
        class_report = classification_report(y_true[col], y_pred[:, i])
        print(col)
        print(class_report)
        

def save_model(model, model_filepath):
    """
    Function to save the save the model
    INPUTS:
        model: the model 
        model_filepath: the file path you would like the model saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


#============================================================
if __name__ == '__main__':
    main()