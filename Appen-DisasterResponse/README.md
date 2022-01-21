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
The first part of your data pipeline is the Extract, Transform, and Load (ETL) process. Please view the python script [process_data.py](https://github.com/Marvykalu/DataEngineering/tree/main/Appen-DisasterResponse/data), you can follow the work through of the ETL pipeline in [ETL_Pipeline.ipynb](https://github.com/Marvykalu/DataEngineering/tree/main/Appen-DisasterResponse/pipeline_notebooks). 


#### Steps
- read each datasets into pandas dataframe. Below is an image of the raw datasets

![Screen Shot 2022-01-21 at 7 56 06 AM](https://user-images.githubusercontent.com/66845704/150539505-db037b07-39b8-4bc1-95fa-b0382d91ca52.png)

- clean the catogories dataset: Below is an image of merged datasets after transformation 

![Screen Shot 2022-01-21 at 8 02 44 AM](https://user-images.githubusercontent.com/66845704/150540299-5c8b6f99-eb37-4bbd-93f2-cdbb90c45e81.png)

To arrive at the transformed datasets, pythons **str.split** method was used to create the 36 individual disaster response categories from the `categories` dataframe as shown in the figure below

![Screen Shot 2022-01-21 at 11 10 25 AM](https://user-images.githubusercontent.com/66845704/150570245-1785719e-d90f-4b6b-aa06-94330484ec87.png)
 
Then we used the first row in the `categories` dataframe to extract a list of new column names for categories. Finally we converted category values to just numbers 0 or 1 by performing indexing and including .str method after the series. Therefore, we resulted at the merged dataframe.


A little data cleaning was done on the combined dataframe such as removing duplicates, columns with only one data value, and filtering dataframe to exclude unwanted data values in some columns.

- store transformed data in an SQLite database. 

To load the data into an SQLite database, we used the pandas dataframe .to_sql() method, with an SQLAlchemy engine (see code snippet below)
![Screen Shot 2022-01-21 at 11 26 54 AM](https://user-images.githubusercontent.com/66845704/150572651-506cb3a1-1bc9-4b5e-b0e8-ff5ffd313072.png)


### Machine Learning pipeline
For the machine learning portion,
- we split the data into a training set and a test set. 
- Then, created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 

-- First we created a tokenize function using NLTK that takes the message (string) and returns a list of tokens. See figure below for how the function works
![Screen Shot 2022-01-21 at 11 29 47 AM](https://user-images.githubusercontent.com/66845704/150573089-4ac83989-df6c-4de8-b197-690804617458.png)

-- Then we created a machine pipeline that takes in the `message` column as input and output classification results on the other 36 categories in the dataset. Since there are multiple target variables, the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) with RandomClassifier was helpful for our prediction. Finally we used GridSearchCVto output a final model.

![Screen Shot 2022-01-21 at 12 17 31 PM](https://user-images.githubusercontent.com/66845704/150579697-6a3297b1-15c9-43a4-871f-119e3f787004.png)

![Screen Shot 2022-01-21 at 12 19 09 PM](https://user-images.githubusercontent.com/66845704/150579715-ff3e13a7-2afc-4948-abc2-46c991a260ee.png)

- Finally, we exported the model to a pickle file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
