# ADA Project - Team HotChocolate
## Beers around our brewtiful world and in our local Satellite 
#### Gabriel Benato, Auriane Debache, Xavier Nal and João Prado

## Abstract 

Beer is an important cultural symbol for many countries and [has generated more than $550bn revenue worldwide in 2021](https://www.statista.com/outlook/cmo/alcoholic-drinks/beer/worldwide#revenue).

Our objective is to tell the story of the greatness of beers and countries through the opinion of beer lovers, while considering statistical corrections of common biases that we, humans, tend to have towards things we enjoy. We aim to use a 'consensus through majority' approach to estimate and correct for the bias of users that are consistently (un)happy with their beverage choices and thus, generate rankings of countries based on their brewed beer's score.

Once this consensus ranking is devised, we propose an excursion to SAT as a last analysis. We wish to provide our peer students the distilled wisdom of the internet and provide some guidance in the quest for the best ale in Ecublens !

## Research Questions
- Which are the best beer brewed in each country, when ratings are corrected with respect to reviewer bias ?
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

First of all, the proportion of NaN values in each column of each dataset was analyzed. Columns that displayed a proportion of NA values going over 60% were discarded, since they would not have provided much information to our analysis. Moreover, it was found that there were some users were present in duplicates in the users dataset. Only one occurrence was kept when this was the case. In addition, breweries that were found to produce no beers that were rated were discarded from the datasets. After exploring the matched beer dataset, it was decided to not use in our analysis, which will consider each one of the main datasets separatedly.

### SAT Dataset processing

We used Vector Space Retrieval based on ```sklearn``` feature extraction module in order to identify SAT beers in the proposed datasets. To this end, we constructed tokenized queries and beer entries based on beer name, brewery name and alcohol content and used cosine distance as a measure of similarity. After a first analysis of our results, matches with cosine similarity smaller than 0.8 were considered not relevant and we kept the best result for each dataset. We checked manually the beers without a match under our similarity threshold and accepeted the low similarity matches that were due to negligable variations.

Beers that were not matched in the dataset were considered to not be rated by any user. We estimate the rating for these beers by implementing a Random Forest Regressor, which learns beer ratings based on features constructed from Alcohol content (ABV), Beer style and Country of origin of the brewery f each beer.

### Exploratory data analysis and first summary statistics

The top 5 location of users and of breweries were computed.
Histogram plots of the number of ratings received by beers and number of ratings ratings per the number of reviews per users and of the number of beers per breweries were also computed.

### Standardization and bias correction


To correct the bias, we propose to apply a correction of systematic reviewer bias, insipired by [this paper](https://krisjensen.github.io/files/bias_blog.pdf). We use a 'mean-field' approach where we correct ratings as such: 

$$ \begin{aligned}r^{\star}_{ij}&= r_{ij} - \alpha_i b_i \\
b_i &= \frac{1}{n_{r,i}} \sum_{\text{All user i ratings}}r_i - \frac{1}{N}\sum_{\text{All ratings}}r 
\end{aligned}$$

Where $r_{ij}$ is the rating of user i to to j, $n_{r,i}$ is the number of ratings given by user $i$, $b_i$ is the estimated bias of user $i$ $\alpha_i$ is an attenuation coefficient specific to user $i$ and based on $n_{r,i}$, $N$ is the total number of ratings in the dataset. 

### Textual analysis of proeminent beers

With the objective of comparing textual reviews of SAT beers appearing in the datasets and with the most proeminent drinks of each dataset, we query the datasets for the most reviewed and favourite beers of users of each country and retrieve reviews that coincide with high ratings (maximum overall rating). We map all reviews to textual embeddings (word vectors) by using OpenAI's Embedding API and ADA-02 model. This choice coomes from the realization that embeddings generated from foundation models are often robust to multilanguage datasets and provide the possibility of comparing our results with other texual corpi, if needed. 

t-SNE representations of these embeddings are produced and we discuss the clusters that were invariant to multiple selections of t-SNE perplexity.

### Happiness 

### Clustering and Ranking

## Organization within the team:

 ### Folder Structure

 - DATA (folder that won't contain the data on git)
   - data's files (Dataset files that will only be present locally)
 - functions (folder)
   - functions' files
 - Plots (folder)
   - plots' files
 - todo_ideas
 - map
   - files for rendering the interactive map


## Contributions of HotChocolate members :

- Auriane :
- Gabriel :
- João : SAT beer pipeline (scrapping, matching data between datasets, visualization, data story)
- Xavier :
