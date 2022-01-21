# Disaster Response Pipeline Project

## Introduction

We hear news of how people are affected by different categories of disasters including earthquakes, storms and poor weather conditions. During such disasters, different disaster response organizations have less time to filter out messages that are important to do their job. However, a model that classifies the disaster messages will be beneficial.

In this project, we applied data engineering skills to analyze thousands of real messages provided by Appen (formally Figure 8) that were sent during natural disasters either via news, social media or directly to disaster response organization.

We identified thirty six (36) categories of the disaster response from the data, and created a machine learning pipeline to categorize future disaster response events so that  the messages can be sent to an appropriate disaster relief agency.



## Objective

The objective of this project was to create and save a multi output supervised ML model for an API that classifies disaster messages. The web app will extract data from database to provide data visualization and use our model to classify new messages for different 36 categories.


## Data

We have two csv data sets and an SQLite database. One disaster_messages.csv contains real messages that were sent during the disaster events. The other disaster_categories.csv contains information about the different disaster response categories. The DisasterResponse.db contains the output from ETL pipleine processes on the two csv files.  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/