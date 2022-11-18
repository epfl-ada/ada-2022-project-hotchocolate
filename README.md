# ADA Project - Team HotChocolate
## Beers around our brewtiful world and in our local Satellite and the Hapiness they bring us
#### Gabriel Benato, Auriane Debache, Xavier Nal and Joao Prado

## Abstract 

Beer is an important cultural symbol for many countries and [has generated more than $550bn revenue worldwide in 2021](https://www.statista.com/outlook/cmo/alcoholic-drinks/beer/worldwide#revenue).

Our objective is to tell the story of the greatness of beers and countries through the opinion and eyes of beer lovers, while considering statistical corrections of common biases that we, humans, tend to have towards things we enjoy. We aim to use a 'consensus through majority' approach to estimate and correct for the bias of users that are consistently (un)happy with their beverage choices and thus, generate rankings of countries based on their brewed beer's score.

Once this consensus ranking is devised, we propose an excursion to SAT as a last analysis. We wish to provide our peer students the distilled wisdom of the internet and provide some guidance in the quest for the best ale in Ecublens !

## Research Questions
- Which are the best beer brewed in each country, when ratings are corrected with respect to reviewer bias ?
- What are the countries and breweries producing the best beers ? Does the result change when we take into account reviewer bias ?
- Does the abv ranking match the beer ranking?
- Does the ranking of countries based on the quality of their beers correlate with the ranking of countries based on happiness?
- What are the best beers sold on SAT ? 
- Is SAT providing good beers to poor students? 
- (Xavier idea, still need to see if feasible) : Is there a correlation between rating and review date when we consider timing of big festivals ?
  
## Proposed additional datasets

We propose an auxiliary dataset consisting of information about 66 beers sold at EPFL's bar, SAT. This datases is constructed by parsing the SAT menu, available [here](https://satellite.bar/bar/). For each beer, the following information is available: 

| Beer name | Price | Type | Brewery name | Origin | Available on tap? (True/False) | Available only seasonally? (True/False) | ABV | Serving volume |
|-----------|-------|------|--------------|--------|------------------|---------------------------|-----|----------------|

We will also extract a countries ranking based on Hapiness in 2017 available at https://allcountries.org/ranks/happiness_index_country_rankings_2017.html.

The following informations will be available:

| Rank | Country/Region | Hapiness' score |
|------|----------------|-----------------|

## Methods

### Data processing

First of all, the proportion of NaN values in each column of each dataset was analyzed. Columns that displayed a proportion of NA values going over a certain threshold (préciser) were discarded, since they would not have provided much information to our analysis. Moreover, it was found that there were some users were present in duplicates in the users dataset. Only one occurrence was kept when this was the case. In addition, breweries that were found to produce no beers that were rated were discarded from the datasets.  

### SAT Dataset processing

We used Vector Space Retrieval based on ```sklearn``` feature extraction module in order to identify SAT beers in the proposed datasets. To this end, we constructed tokenized queries and beer entries based on beer name, brewery name and alcohol content and used cosine distance as a measure of similarity. After a first analysis of our results, matches with cosine similarity smaller than 0.7 were considered not relevant and we keep the best result for each dataset. We checked manually the beers without a match under our similarity threshold and accepeted the low similarity matches that wwere due to negligable variations.

### Exploratory data analysis and first summary statistics

The top 5 location of users and of breweries were computed.
Histogram plots of the number of reviews per users and of the number of beers per breweries were also computed.

### Standardization and bias correction


To correct the bias, we propose to compute the median rating  for each beer. For each beer an user will have rated, we will also compute the difference between their rating and the median rating of the beer. We will then average these differences.

$$\text{UserCorrection} = \frac{\displaystyle\sum_{UserRatings}{\frac{BeerScore_{median} - BeerScore_{user}}{\sigma_{BeerScore}}}}{N_{UserRatings}}$$

This correction will be added to all the user's ratings. Moreover, when we compute averages, more weight will be given to users which number of reviews go over a certain threshold. 

### Textual and rating analysis

A first exploration of the textual reviews and ratings has been performed by computing summary statistics and histograms of word counts per review and date of creation per rating/review. We then used a ```fasttext``` pretrained model in order to identify the most common languages in the reviews of both BeerAdvocate and RateBeer. A decision was taken to only consider ratings in our future analysis, since the heterogeneity of textual reviews, (in terms of size, content and language) make their study non trivial. 

## Proposed timeline

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

- Notebook.
- data (folder that won't contain the data on git)
  - data's files (Dataset files that will only be present locally)
- functions (folder)
  - functions' files
- Plots (folder)
  - plots' files
- todo_ideas
  
### TODOs
- Read ```todo_ideas```.

### Notes for group
Install in developer mode with `pip install -e .` from a terminal in the project's main folder.

Here is the Google DRive link to download the data 
https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF
