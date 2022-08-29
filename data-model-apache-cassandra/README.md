# Data Modeling with Apache Cassandra

## Introduction

A non-relational database was created for a music streaming app using Apache Cassandra. The purpose of the NoSQL database is to answer queries on song play data. There were three queries in total, thus the data model includes a table for each of the following queries:

### Queries

- 1) Give me the artist, song title and song's length in the music app history that was heard during sessionId = 338, and itemInSession = 4

- 2) Give me only the following: name of artist, song (sorted by itemInSession) and user (first and last name) for userid = 10, sessionid = 182

- 3) Give me every user name (first and last) in my music app history who listened to the song 'All Hands Against His Own'

## project-cassandra

The jupyter notebook file `project-cassandra` contains ETL pipeline and data modeling steps.

#### ETL Pipeline: Data preprocessing

- The data is stored as a collection of csv files partitioned by date. In this step, I processed the event_datafile_new.csv dataset to create a denormalized dataset as shown below.

- The ETL pipeline basically iterates through each event file in event_data to process and create a new CSV file which is used to populate the denormalized database optimised for the 3 queries above. 



from cassandra.cluster import Cassandra

cluster = Cluster(['127.0.0.1'])
session = cluster.connect()
To stop and remove the container after the exercise

docker stop cassandra-container
docker rm cassandra-container