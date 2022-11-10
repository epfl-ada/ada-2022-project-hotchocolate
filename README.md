# ADA Project - Team HotChocolate
#### Gabriel Benato, Auriane Debache, Xavier Nal and Joao Prado

## Introduction 

## Research Questions
  
## Proposed additional datasets

We propose an auxiliary dataset consisting of information about 72 beers sold at EPFL's bar, SAT. This datases is constructed by parsing the SAT menu, available here. For each beer, the following information is available: 

| Beer name | Price | Type | Brewery name | Origin | Available on tap? (True/False) | Available only seasonally? (True/False) | ABV | Serving volume |
|-----------|-------|------|--------------|--------|------------------|---------------------------|-----|----------------|

We use ```nltk```,```sklearn.feature_extraction``` and our own flavor of vector space retrieval to identify these beers in BeerAdvocate and RateBeer datasets. A full analysis is available in ```notebooks/SAT beers.ipynb```.

## Methods

## Proposed timeline

### Before Week 47 (Now)

Milestone 2 : 
- Top 10 beers by country, by type, for each site separatedly. (without bias correction)
- Histogram of users in terms of their rating counts (already partially done by Gab).
- Top X breweries (in terms of beer count, and how many ratings their beer have). 
- Top 10 beers in matched dataset (without bias correction),
- Basic textual review analysis (% of language, review length)
- Wordcloud with most commonly used words (if time allows)


### Week 47 (21/11-25/11)
Homework 2
### Week 48 (28/11-2/12)
Homework 2

First implementation of the website (1 week, João)

Design a simple regression model to predict beer rating (1 week, 1 person)
### Week 49 (5/12-9/12)


Finish interactive map chart. (1 week, Gabriel)

Bias correction for data analysis (3 days, Auriane)

Statistical analysis (4 day, everyone)


### Week 50 (12/12-16/12)

Write down our conclusions (only once two last steps are validated) (4 days, 2 persons) 

Finish all plots (4 days, 1 person)

Correct text (1 day, everyone)

Design (3 days, Auriane)

### Week 51 (19/12-23/12)
Safety net week - We hope to be finished before 19/12 and to use this week only in case of an emergency

__Landmark__ : A brewtiful data analysis is live !

## Organization within the team:

### Set-up

- main
- data (folder that won't contain the data on git)
  - data's files (Dataset files that will only be present locally)
- functions (folder)
  - functions' files
- Plots (folder)
  - plots' files 
  
### TODOs
- Read ```todo_ideas```.

### Notes for group
Install in developer mode with `pip install -e .` from a terminal in the project's main folder.

Here is the Google DRive link to download the data 
https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF
