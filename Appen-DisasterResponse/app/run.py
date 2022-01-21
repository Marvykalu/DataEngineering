import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_disaster_response', engine)

# load model
model = joblib.load("../models/ML_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    genre_counts = df['genre'].value_counts()
    genre_names = list(genre_counts.index)

    # Show distribution of top 10 disaster response categories
    category = df.iloc[:,4:]
    category_total = category.sum().sort_values(ascending = False)
    category_names = list(category_total.index)


    # Show distribution of top 10 disaster response categories
    #with respect to message genre ('news','direct','social')
    genre_news_df = df[df['genre'] == 'news'].iloc[:,4:]
    genre_news_prop = genre_news_df.mean().sort_values(ascending = False)[1:11]
    genre_news_names = genre_news_prop.index

    genre_direct_df = df[df['genre'] == 'news'].iloc[:,4:]
    genre_direct_prop = genre_direct_df.mean().sort_values(ascending = False)[1:11]
    genre_direct_names = genre_direct_prop.index

    genre_social_df = df[df['genre'] == 'news'].iloc[:,4:]
    genre_social_prop = genre_social_df.mean().sort_values(ascending = False)[1:11]
    genre_social_names = genre_social_prop.index



    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_total
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Disaster Response Categories',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_news_names,
                    y=genre_news_prop
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Disaster Response Categories sent via News',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_direct_names,
                    y=genre_direct_prop
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Disaster Response Categories sent directly to Disaster Response Organization',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_social_names,
                    y=genre_social_prop
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Disaster Response Categories sent via Socials',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
