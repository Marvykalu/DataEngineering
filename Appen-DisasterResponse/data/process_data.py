# Project: Building a Disaster Response Pipeline

"""
A python script that shows the Extract, Transform and Load Pipeline processes:

- it takes the file paths of the two datasets and database, 
- cleans the datasets, and 
- stores the clean data into a SQLite database in the specified database file path.

"""
#================================================================

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to extract, transform and load given datasets
    
    INPUT:
        messages_filepath: The path of messages dataset; e.g 'messages.csv'
        categories_filepath: The path of categories dataset e.g 'categories.csv'
    OUTPUT:
        df: A merged dataset of messages_filepath and categories_filepath
    """
    # load dataset
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    
    # merge datasets on common column features
    df = messages.merge(categories, on='id', how='inner')
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    
    # select the first row of the categories dataframe,
    # use this row to extract a list of new column names for categories dataset
    #rename the columns of categories dataset
    
    row = categories.iloc[0] #first row of categories dataframe
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist() #gives a list of strings
    categories.columns = category_colnames #rename the columns of `categories`
    
    # Convert columns data values in categories to just numbers 0 or 1.
    
    for column in categories:
        # convert column to string and extract last character of string
        categories[column] = categories[column].astype(str).str[-1:] #gives string values e.g '0', '1'
        # convert column values from string to numeric
        categories[column] = categories[column].astype(int) #gives numeric values e.g 0,1
        
    
    # drop the original categories column from `df` dataframe
    df.drop('categories', axis=1, inplace = True)
    # concatenate the `df` dataframe with the transformed `categories` dataframe
    df = pd.concat([df, categories], axis=1) #yields a dataframe of shape (26386, 40)
    return df



def clean_data(df):
    """
    Function to perform relevant cleaning on the given dataframe
    
    INPUT:
        df: Given raw dataframe from load_data() function in previous step
    OUTPUT:
        df: cleaned dataframe
    """
    # drop duplicates, irrelevant columns and filter dataframe
    
    df = df.drop_duplicates() #drop duplicates
    df.drop('child_alone', axis = 1, inplace = True) #drop `child_alone` column with only one data value '0'
    df = df.query('related != 2') #filter df to exclude values of 2 from `related` column
    
    return df


def save_data(df, database_filename):
    """
    Function to save the transformed (cleaned) dataframe from previous step to SQLite database
    
    INPUT:
        df: transformed dataframe
        database_filename: give the name of your database e.g 'DisasterResponse.db'
    OUTPUT:
        A SQL database with table `message_disaster_response`
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_disaster_response', engine,if_exists = 'replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
 
#===================================================================

if __name__ == '__main__':
    main()