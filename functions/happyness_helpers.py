#!/usr/bin/python3


"""
This file pool all the helpers functions used for positives and negatives words analysis in the review

"""
import pandas as pd
import numpy as np
import requests
import re



def loop_AdvBeer_get_pos_neg_words():
    """
    automatic loop throuhg all advocate beer csv to do an postive and negative words analysis
    This function will get the mean of positifs and negatifs per countries by matching the user-id with the location
    in the user_id dataset from Advocate Beer.
    (disclaimer this function can take multiple hours to run, it need to run thought 12 csv of 250 000 data of reviews)

    input: list of positifs and negatif words, csv files in the correct folder 'DATA/BeerAdvocate/'
    output: csv with positif and negatifs words average per country's review
    """
    csv_count = 0

    advBeer_root = 'DATA/BeerAdvocate/'
    generated_root = 'generated/'
    df_adv_users = pd.read_csv(advBeer_root + 'users.csv')
    
    # get list of positif and negatif words
    positive_word_document = requests.get("https://ptrckprry.com/course/ssd/data/positive-words.txt")
    negative_word_document = requests.get("https://ptrckprry.com/course/ssd/data/negative-words.txt")

    positive_words = re.search(r'(?<=;[\r\n]{2}).*',positive_word_document.text,re.DOTALL).group(0).split()
    negative_words = re.search(r'(?<=;[\r\n]{2}).*',negative_word_document.text,re.DOTALL).group(0).split()

    #dictionary of sentiment words
    sentiment_dict = {
        "positive" : positive_words,
        "negative" : negative_words
    }


    #loop over all sub .csv dataset (each dataset have max 250 000 rows)
    for csv_count in range(0,11):
        print('csv part:',csv_count)
        df_rev_low = pd.read_csv(advBeer_root + 'BeerAdvocate_reviews_part_' + str(csv_count) +'.csv')
        df_rev_low = df_rev_low[['beer_id', 'brewery_id', 'user_id', 'text']]
        list_user_location_adv = df_adv_users.loc[df_adv_users['user_id'].isin(df_rev_low['user_id']),['user_id','location']]
        print('shape:',df_rev_low.shape, 'list user location adv shape', list_user_location_adv.shape)

        df_rev_low.loc[:,'location'] = df_rev_low['user_id'].apply(get_location_user, args= (list_user_location_adv,))

        df_rev_low.to_csv(generated_root + 'BeerAdvocate_location_user_id_part_' + str(csv_count) +'.csv')

        print('location done for part:', csv_count)

        #group by country
        country_groups = df_rev_low.groupby(by="location")

        # Positive, Negatifs words analysis
        data = []

        # loop for to compute the number of pos and neg words for each country
        for country, group in country_groups:
            negative_words = []
            positive_words = []
            print('country:', country, 'size:', group.shape[0])

            negative_words.append(group.apply(lambda row : indicator_words('negative',row['text'],sentiment_dict),axis=1))
            positive_words.append(group.apply(lambda row : indicator_words('positive',row['text'],sentiment_dict),axis=1))


            data.append([country,np.mean(negative_words),np.mean(positive_words), group.shape[0]])
            #resultats_pos_neg_words_2 = pd.DataFrame(data, columns=["country","neg_words", "pos_words", "nb_review"])


        resultats_pos_neg_words = pd.DataFrame(data, columns=["country","neg_words", "pos_words", "nb_review"])

        #save the data
        resultats_pos_neg_words.to_csv(generated_root + 'BeerAdvocate_pos_neg_words_analysis_part_' + str(csv_count) +'.csv')
        print('--------------------------------------------------------')
        
        return
        

def loop_RateBeer_get_pos_neg_words():
    """
    automatic loop throuhg all rate beer csv to do an postive and negative words analysis
    This function will get the mean of positifs and negatifs per countries by matching the user-id with the location
    in the user_id dataset from RateBeer.
    (disclaimer this function can take multiple hours to run, it need to run thought 28 csv of 250 000 data of reviews)
    
    input: list of positifs and negatif words, csv files in the correct folder 'DATA/RateBeer/'
    output: csv with positif and negatifs words average per country's review
    """
    csv_count = 0
    
    rateBeer_root = 'DATA/RateBeer/'
    generated_root = 'generated/'
    df_rate_users = pd.read_csv(rateBeer_root + 'users.csv')
    
    # get list of positif and negatif words
    positive_word_document = requests.get("https://ptrckprry.com/course/ssd/data/positive-words.txt")
    negative_word_document = requests.get("https://ptrckprry.com/course/ssd/data/negative-words.txt")

    positive_words = re.search(r'(?<=;[\r\n]{2}).*',positive_word_document.text,re.DOTALL).group(0).split()
    negative_words = re.search(r'(?<=;[\r\n]{2}).*',negative_word_document.text,re.DOTALL).group(0).split()

    #dictionary of sentiment words
    sentiment_dict = {
        "positive" : positive_words,
        "negative" : negative_words
    }
    


    #loop over all sub .csv dataset (each dataset have max 250 000 rows)
    for csv_count in range(2,29):
        print('csv part:',csv_count)
        df_rev_low = pd.read_csv(rateBeer_root + 'RateBeer_reviews_part_' + str(csv_count) +'.csv')
        df_rev_low = df_rev_low[['beer_id', 'brewery_id', 'user_id', 'text']]
        list_user_location = df_rate_users.loc[df_rate_users['user_id'].isin(df_rev_low['user_id']),['user_id','location']]
        print('shape:',df_rev_low.shape, 'list user location shape', list_user_location.shape)

        df_rev_low.loc[:,'location'] = df_rev_low['user_id'].apply(get_location_user, args= (list_user_location,))

        df_rev_low.to_csv(generated_root + 'RateBeer_location_user_id_part_' + str(csv_count) +'.csv')

        print('location done for part:', csv_count)

        #group by country
        country_groups = df_rev_low.groupby(by="location")

        # Positive, Negatifs words analysis
        data = []

        # loop for to compute the number of pos and neg words for each country
        for country, group in country_groups:
            negative_words = []
            positive_words = []
            print('country:', country, 'size:', group.shape[0])

            negative_words.append(group.apply(lambda row : indicator_words('negative',row['text'],sentiment_dict),axis=1))
            positive_words.append(group.apply(lambda row : indicator_words('positive',row['text'],sentiment_dict),axis=1))


            data.append([country,np.mean(negative_words),np.mean(positive_words), group.shape[0]])
            #resultats_pos_neg_words_2 = pd.DataFrame(data, columns=["country","neg_words", "pos_words", "nb_review"])


        resultats_pos_neg_words = pd.DataFrame(data, columns=["country","neg_words", "pos_words", "nb_review"])

        #save the data
        resultats_pos_neg_words.to_csv(generated_root + 'RateBeer_pos_neg_words_analysis_part_' + str(csv_count) +'.csv', index=False)
        print('--------------------------------------------------------')

    return

def get_location_user(x, review):
    """
    This function return the location of a user with using his/her location in the review list
    input: x (user_id, type 'str'), review (dataframe with columns:user_id, location, dtypes object)
    output: location value of the user_id (type: np.array.values, value type 'str')
    """
    #print(x, type(x))
    location =  review.loc[review['user_id'] == x,'location'].values 
    # print('loca', type(location), location, x)
    
    # if user id is not in the list ( like: 461238)
    if not len(location):
        location = ['nan']
    
    # location is a np.array, we just return the value in this array    
    return location[0]


def indicator_words(words,review,dict):
    """An indicator function which outputs the numbers of words of the list words present in the review  and 0 if not

    Parameters
    ----------
    words : str
        Words corresponding to the indicator function. 
    review : str
        The review to be tested by the indicator function. 

    Returns
    -------
    int
        number of word present of the list of positifs or negatifs words
    """
    #convert to string to avoid problem
    review = str(review)
    #We filter the review with regex and apply lower since the function is agnostic to case and punctuation.
    #The re function re.sub() substitutes all matches of the expression in its first argument by its second argument
    #The regex expression [!?.]* matches characters '!', '?' and '.' an indefinite amount of times. 
    review_words = re.sub(r'[!?.]*','',review).lower().split()
    #res = int(any([word in dict[words] for word in headline_words]))
    res = np.sum([word in dict[words] for word in review_words])
    #print('res', res)
    return res

# compute the weighted average for positive and negative review for a group
def weighted_average_all(dataframe, pos, neg, weight):
    #print(pos, type(value), weight)
    #print(dataframe.shape)
    #print(dataframe)
    valp = dataframe[pos]
    valn = dataframe[neg]
    #print(valp)
    wt = dataframe[weight]
    return (valn * wt).sum() / wt.sum(), (valp * wt).sum() / wt.sum(),wt.sum()


def ratebeer_merging_csv_results():
    """
    This function managed to concatenate all csv resutls of RateBeer positive and negative words analysis
    output: dataframe with all results concatenate
    output columns are: 'country','neg_words','pos_words','nb_review'
    output rows: country but all country data are not merge at this point
    """
    generated_root = 'generated/'
    
    # get all the results from all sub .csv dataset
    df_rb_part_0 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_0.csv')
    df_rb_part_1 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_1.csv')
    df_rb_part_2 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_2.csv')
    df_rb_part_3 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_3.csv')
    df_rb_part_4 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_4.csv')
    df_rb_part_5 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_5.csv')
    df_rb_part_6 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_6.csv')
    df_rb_part_7 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_7.csv')
    df_rb_part_8 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_8.csv')
    df_rb_part_9 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_9.csv')
    df_rb_part_10 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_10.csv')
    df_rb_part_11 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_11.csv')
    df_rb_part_12 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_12.csv')
    df_rb_part_13 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_13.csv')
    df_rb_part_14 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_14.csv')
    df_rb_part_15 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_15.csv')
    df_rb_part_16 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_16.csv')
    df_rb_part_17 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_17.csv')
    df_rb_part_18 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_18.csv')
    df_rb_part_19 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_19.csv')
    df_rb_part_20 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_20.csv')
    df_rb_part_21 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_21.csv')
    df_rb_part_22 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_22.csv')
    df_rb_part_23 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_23.csv')
    df_rb_part_24 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_24.csv')
    df_rb_part_25 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_25.csv')
    df_rb_part_26 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_26.csv')
    df_rb_part_27 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_27.csv')
    df_rb_part_28 = pd.read_csv(generated_root +  'RateBeer_pos_neg_words_analysis_part_28.csv')
    
    #One dataset of RateBeer to rule them all, One dataset to find them, 
    #One dataset to bring them all and in the darkness bind them 
    
    # concat all the dataframe to group them by country after for futur analysis
    df_all_rb = pd.concat([df_rb_part_0, df_rb_part_1, df_rb_part_3, df_rb_part_4, df_rb_part_5, df_rb_part_6, df_rb_part_7,
                       df_rb_part_8, df_rb_part_9,df_rb_part_10, df_rb_part_11, df_rb_part_12, df_rb_part_13,df_rb_part_14,
                       df_rb_part_15, df_rb_part_16, df_rb_part_17, df_rb_part_18, df_rb_part_19, df_rb_part_20,
                       df_rb_part_21, df_rb_part_22, df_rb_part_23, df_rb_part_24, df_rb_part_25, df_rb_part_26,
                       df_rb_part_27, df_rb_part_28], ignore_index=True)
    df_all_rb = df_all_rb[['country','neg_words','pos_words','nb_review']]
    
    return df_all_rb


def advbeer_merging_csv_results():
    generated_root = 'generated/'
    
    # get all the results from all sub .csv dataset
    df_part_0 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_0.csv')
    df_part_1 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_1.csv')
    df_part_2 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_2.csv')
    df_part_3 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_3.csv')
    df_part_4 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_4.csv')
    df_part_5 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_5.csv')
    df_part_6 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_6.csv')
    df_part_7 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_7.csv')
    df_part_8 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_8.csv')
    df_part_9 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_9.csv')
    df_part_10 = pd.read_csv(generated_root +  'BeerAdvocate_pos_neg_words_analysis_part_10.csv')
    
    # One dataset to rule them all, One dataset to find them, 
    # One dataset to bring them all and in the darkness bind them 
    
    # concat all the dataframe to group them by country after for futur analysis
    df_all = pd.concat([df_part_0, df_part_1, df_part_3, df_part_4, df_part_5, df_part_6, df_part_7,
                       df_part_8, df_part_9,df_part_10], ignore_index=True)
    df_all = df_all[['country','neg_words','pos_words','nb_review']]
    
    return df_all



# old helpers functions

def indicator_words(words,review,dict):
    """An indicator function which outputs the numbers of words of the list words present in the review  and 0 if not

    Parameters
    ----------
    words : str
        Words corresponding to the indicator function. 
    review : str
        The review to be tested by the indicator function. 

    Returns
    -------
    int
        number of word present of the list of positifs or negatifs words
    """
    #We filter the review with regex and apply lower since the function is agnostic to case and punctuation.
    #The re function re.sub() substitutes all matches of the expression in its first argument by its second argument
    #The regex expression [!?.]* matches characters '!', '?' and '.' an indefinite amount of times. 
    review_words = re.sub(r'[!?.]*','',review).lower().split()
    #res = int(any([word in dict[words] for word in headline_words]))
    res = np.sum([word in dict[words] for word in review_words])
    #print('res', res)
    return res

def weighted_average(dataframe, value, weight):
    print(value, type(value), weight)
    print(dataframe.shape)
    print(dataframe)
    val = dataframe[value]
    print(val)
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()



# took from hw1
def indicator(words,headline,dict):
    """An indicator function which outputs 1 if the headline has at least one word of type pronoun_type and 0 if not

    Parameters
    ----------
    pronoun_type : str
        The pronoun type corresponding to the indicator function. Should match exactly a key of feature_wordsets
    headline : str
        The headline to be tested by the indicator function. 

    Returns
    -------
    int
        1 if the headline has at least one word of type pronoun_type and 0 if not
    """
    #We filter the headline with regex and apply lower since the function is agnostic to case and punctuation.
    #The re function re.sub() substitutes all matches of the expression in its first argument by its second argument
    #The regex expression [!?.]* matches characters '!', '?' and '.' an indefinite amount of times. 
    headline_words = re.sub(r'[!?.]*','',headline).lower().split()
    res = int(any([word in dict[words] for word in headline_words]))
    #print('res', res)
    return res