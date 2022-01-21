# Disaster Response Pipeline Project

## Introduction

We hear news of how people are affected by different categories of disasters including earthquakes, storms and poor weather conditions. During such disasters, different disaster response organizations have less time to filter out messages that are important to do their job. However, a model that classifies the disaster messages will be beneficial.

In this project, we applied data engineering skills to analyze thousands of real messages provided by Appen (formally Figure 8) that were sent during natural disasters either via news, social media or directly to disaster response organization.

We identified thirty six (36) categories of the disaster response from the data, and created a machine learning pipeline to categorize future disaster response events so that  the messages can be sent to an appropriate disaster relief agency.



## Objective

The objective of this project was to create and save a multi output supervised ML model for an API that classifies disaster messages. The web app will extract data from database to provide data visualization and use our model to classify new messages for different 36 categories.


## Data

We have two csv data sets: One disaster_messages.csv contains real messages that were sent during the disaster events, the other disaster_categories.csv contains information about the different disaster response categories. These two datasets were processed through an ETL pipeline and the output of the pipeline was an SQLite database (DisasterResponse.db). 

## Methodology
Here we will give you details of the steps during the data processing and modelling 

### ETL pipeline
The first part of your data pipeline is the Extract, Transform, and Load process. Please view the python script [process_data.py](https://github.com/Marvykalu/DataEngineering/tree/main/Appen-DisasterResponse/data), you can follow the work through of the ETL pipeline in [ETL_Pipeline.ipynb](https://github.com/Marvykalu/DataEngineering/tree/main/Appen-DisasterResponse/pipeline_notebooks). 


#### Steps
- read each datasets into pandas dataframe
- clean the catogories dataset, and then store it in a SQLite database. We expect you to do the data cleaning with pandas. To load the data into an SQLite database, you can use the pandas dataframe .to_sql() method, which you can use with an SQLAlchemy engine.
![Screen Shot 2022-01-21 at 7 56 06 AM](https://user-images.githubusercontent.com/66845704/150539505-db037b07-39b8-4bc1-95fa-b0382d91ca52.png)



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
