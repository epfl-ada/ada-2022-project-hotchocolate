# ADA Project - Team HotChocolate
## Beers around our brewtiful world and in our local Satellite 
#### Gabriel Benato, Auriane Debache, Xavier Nal and João Prado

## Abstract 

Beer is an important cultural symbol for many countries and [has generated more than $550bn revenue worldwide in 2021](https://www.statista.com/outlook/cmo/alcoholic-drinks/beer/worldwide#revenue).

Our objective is to tell the story of the greatness of beers and countries through the opinion of beer lovers, while considering statistical corrections of common biases that we, humans, tend to have towards things we enjoy. We aim to use a 'consensus through majority' approach to estimate and correct for the bias of users that are consistently (un)happy with their beverage choices and thus, generate rankings of countries based on their brewed beer's score.

Once this consensus ranking is devised, we propose an excursion to SAT as a last analysis. We wish to provide our peer students the distilled wisdom of the internet and provide some guidance in the quest for the best ale in Ecublens !

## Research Questions

- Can we provide a satisfying correction of systematic user bias?
- Which beers are preferred by different nationalities? 
- Do some nationalities have better taste than others?
- Are some nationalities more prone to praise in their reviews than other nationalities?
- Can we predict the ratings of beers sold at SAT but not found in our datasets?
- What is the rating ranking of beers sold at SAT?
- Is this rating correlated to the price and volume of the beers?
- Can we provide a menu of beers sold at SAT for each nationality? 
- Can we provide a description of all beers sold at SAT to help customers to make their choice?


- What are the countries and breweries producing the best beers ? Does the result change when we take into account reviewer bias ?
- Does the alcohol by volume ranking match the beer ranking?
- Does the ranking of countries based on the quality of their beers correlate with the ranking of countries based on happiness?
- What are the best beers sold on SAT ? 
- Is SAT providing good beers to poor students? 
-Addtional question if we have enough time: can we predict the ratings that the beers sold at SAT but not found in our datasets?
  
## Proposed additional datasets

We propose an auxiliary dataset consisting of information about 66 beers sold at EPFL's bar, SAT. This datases is constructed by parsing the SAT menu, available [here](https://satellite.bar/bar/). For each beer, the following information is available: 

| Beer name | Price | Type | Brewery name | Origin | Available on tap? (True/False) | Available only seasonally? (True/False) | ABV | Serving volume |
|-----------|-------|------|--------------|--------|------------------|---------------------------|-----|----------------|

We will also extract a countries ranking based on Hapiness in 2017 available at https://allcountries.org/ranks/happiness_index_country_rankings_2017.html.

The following information will be available:

| Rank | Country/Region | Hapiness' score |
|------|----------------|-----------------|

## Methods

### Data processing

First of all, the proportion of NaN values in each column of each dataset was analyzed. Columns that displayed a proportion of NA values going over 60% were discarded, since they would not have provided much information to our analysis. Beers that did not display any ratings were removed. Moreover, it was found that there were some users were present in duplicates in the users dataset. Only one occurrence was kept when this was the case. In addition, breweries that were found to produce no beers that were rated were discarded from the datasets. After exploring the matched beer dataset, it was decided to not use in our analysis, which will consider each one of the main datasets separatedly.

### SAT Dataset processing

We used Vector Space Retrieval in order to identify SAT beers in the proposed datasets. To this end, we constructed tokenized queries and beer entries based on beer name, brewery name and alcohol content and used cosine distance as a measure of similarity. After a first analysis of our results, matches with cosine similarity smaller than 0.8 were considered not relevant and we kept the best result for each dataset. We checked manually the beers without a match under our similarity threshold and accepeted the low similarity matches that were due to negligable variations.

Beers that were not matched in the dataset were considered to not be rated by any user. We estimate the rating for these beers by implementing a Random Forest Regressor, which learns beer ratings based on features constructed from Alcohol content (ABV), Beer style and Country of origin of the brewery f each beer.


### Standardization and bias correction


To correct the bias, we propose to apply a correction of systematic reviewer bias, insipired by [this paper](https://krisjensen.github.io/files/bias_blog.pdf). We use a 'mean-field' approach where we correct ratings as such: 

$$ \begin{aligned}r^{\star}_{ij}&= r_{ij} - \alpha_i b_i \\
b_i &= \frac{1}{n_{r,i}} \sum_{\text{All user i ratings}}r_i - \frac{1}{N}\sum_{\text{All ratings}}r 
\end{aligned}$$

Where $r_{ij}$ is the rating of user i for beer j, $n_{r,i}$ is the number of ratings given by user $i$, $b_i$ is the estimated bias of user $i$ $\alpha_i$ is an attenuation coefficient specific to user $i$ and based on $n_{r,i}$, $N$ is the total number of ratings in the dataset. 

The bias of user i is attenuated by attenuation coefficient $\alpha_i$ computed as follows:

$$ \begin{aligned} n_{r,i}=1 :  \alpha_i=0 
1<n_{r,i}<n_{max}: \alpha_i=\frac{1}{n_{max}-1}n_{r,i}-\frac{1}{n_{max}-1} 
n_{r,i}=n_{max}; \alpha_i=1 \end{aligned}$$
where $n_{max}$ is the maximum number of ratings given by an user. This allows to give cancel out biases of one-time reviewers, give maximum weight to the user that gave $n_{max}$ reviews, and to increase proportionally the bias of other reviewers with their $n_{r,i}$.

### Textual analysis of proeminent beers

With the objective of comparing textual reviews of SAT beers appearing in the datasets and with the most proeminent drinks of each dataset, we query the datasets for the most reviewed and favourite beers of users of each country and retrieve reviews that coincide with high ratings (maximum overall rating). We map all reviews to textual embeddings (word vectors) by using OpenAI's Embedding API and ADA-02 model. This choice coomes from the realization that embeddings generated from foundation models are often robust to multilanguage datasets and provide the possibility of comparing our results with other texual corpi, if needed. 

t-SNE representations of these embeddings are produced and we discuss the clusters that were invariant to multiple selections of t-SNE perplexity.

### Happiness 

The happiness analysis began by using sentiment analysis, similar to what we learned in class. However, it was found that most of the reviews had a neutral sentiment and there was not a clear negative or positive sentiment in the results for each country's review. This can be explained by the fact that the reviews were mainly descriptive.

Therefore, we decided to look at the number of positive and negative words in the reviews and group them by the country of the user. To do this, we used the list of positive and negative words provided in the course for homework 1. The mean frequency of these words was calculated for the two datasets, RateBeer and AdvocateBeer. This analysis required separating the reviews into several CSV files in order to run the algorithm correctly on the Jupyter notebook without overloading the memory. This analysis has been added to the world map and gives users a new metric to compare the beer of different countries and see which beer elicits the most enthusiasm from users.

### Clustering and Ranking

## Organization within the team:

 ### Folder Structure

 - DATA (folder that won't contain the data on git)
   - data's files (Dataset files that will only be present locally)
 - functions (folder)
   - functions' files
 - Plots (folder)
   - plots' files
 - generated (folder with csv data generated for the postif and negatifs words analysis)
 - todo_ideas
 - map
   - files for rendering the interactive map


## Contributions of HotChocolate members :

- Auriane :
- Gabriel : Interactive map, code cleaning and data story
- João : SAT beer pipeline (scrapping, matching data between datasets, visualization, data story)
- Xavier : Sentiment Analysis with nlp, Positive and negative words for all reviews analysis
