#!/usr/bin/python3
import json
import re
import string
from collections import Counter
from operator import itemgetter
import numpy as np
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import fasttext
import warnings
#Do not want warnings about nltk libraries
warnings.filterwarnings("ignore")
## Helper functions for Information Retrieval Natural Language Processing on datasets
tfidf = TfidfVectorizer()

nltk.download('stopwords')
def vector_space_retrieval(queries,dataframe,k=1):
    document_dict = tokenize_dataframe(dataframe)
    doc_vectors = tfidf.fit_transform(document_dict.values())
    doc_ids = []
    similarity_coefs = []
    for query in queries.values():
        vector_queries = tfidf.transform([query])
        cosine_similarities = linear_kernel(vector_queries, doc_vectors).flatten()
        related_docs_indices, cos_sim_sorted = zip(*sorted(enumerate(cosine_similarities), key=itemgetter(1), 
                                                        reverse=True))
        for i, cos_sim in enumerate(cos_sim_sorted):
            if i >= k:
                break
            doc_ids.append(related_docs_indices[i])
            similarity_coefs.append(cos_sim)
    new_df = dataframe.iloc[doc_ids].copy()
    new_df["similarity"] = np.array(similarity_coefs)
    return new_df

def tokenize(text):
    stemmer = PorterStemmer()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word.lower()) for word in tokens])

def tokenize_dataframe(dataframe):
    documents = {}
    for index, beer in dataframe.iterrows():
        documents[beer["beer_id"]] = (tokenize(beer["beer_name"]) + " " + tokenize(beer['brewery_name']) + " " + tokenize(str(beer["abv"])))
    print("Documents tokenized")
    return documents

def search_vec(query, features, threshold=0.1):
    new_features = tfidf.transform([query])
    cosine_similarities = linear_kernel(new_features, features).flatten()
    related_docs_indices, cos_sim_sorted = zip(*sorted(enumerate(cosine_similarities), key=itemgetter(1), 
                                                       reverse=True))
    doc_ids = []
    for i, cos_sim in enumerate(cos_sim_sorted):
        if cos_sim < threshold:
            break
        doc_ids.append(related_docs_indices[i])
    return doc_ids

def color(df,similarity,threshold=0.6):
    good = f"background-color:#C6F8D3" 
    bad = f"background-color:#F8C6C6" 
    color_df = pd.DataFrame('',index=df.index,columns=df.columns)
    BA_good_fits = similarity[similarity["BA_similarity"] > threshold ].index
    RB_good_fits = similarity[similarity["RB_similarity"] > threshold ].index
    BA_bad_fits = similarity[similarity["BA_similarity"] <= threshold ].index
    RB_bad_fits = similarity[similarity["RB_similarity"] <= threshold ].index
    color_df.loc[BA_good_fits,'BA_beer_name'] = good
    color_df.loc[RB_good_fits,'RB_beer_name'] = good
    color_df.loc[BA_bad_fits,'BA_beer_name'] = bad
    color_df.loc[RB_bad_fits,'RB_beer_name'] = bad
    return color_df

def querify(query_df):
    sat_queries = {}
    for index, biere in query_df.iterrows():
        sat_queries[index] = (tokenize(biere['nom']) +" "  + tokenize(biere['brasseur']) + " " + tokenize(str(biere['alcool'])))
    return sat_queries

def display_results_df(base_df,results_df,reference_column,results):
    new_df = pd.DataFrame()
    new_df[reference_column] = base_df[reference_column]
    new_df[results] = results_df[results].copy()
    return new_df

### Helper functions for word count, language identification and date 
def fetch_ratings(dataset):
    first = pd.read_csv(f"data/{dataset}_ratings_part_0.csv",low_memory=False)
    if dataset == "BeerAdvocate" :
        first = first[first['rating'] != ' nan']
    if dataset == "RateBeer" :
        first = first[first['rating'] != 'NaN']    
    ratings = first.rating.astype(float)
    if dataset == "BeerAdvocate" :
        csv_count = 17
    else :
        csv_count = 15
    for index in range(1,csv_count):   
        temp = pd.read_csv(f"data/{dataset}_ratings_part_{index}.csv",low_memory=False)
        if dataset == "BeerAdvocate" :
            first = temp[temp['rating'] != ' nan']
        if dataset == "RateBeer" :
            first = temp[temp['rating'] != 'NaN']        
        
        rating = temp.rating.astype(float)
        ratings = dates = pd.concat([ratings, rating])
    return ratings

def summary_analysis(dataset):
    first = pd.read_csv(f"data/{dataset}_ratings_part_0.csv",low_memory=False)
    #TODO: standardize NaNs between datasets
    if dataset == "BeerAdvocate" :
        first = first[first['text'] != ' nan']
    if dataset == "RateBeer" :
        first = first[first['text'] != 'NaN']
    print("Started identifying languages, counting words and binning dates...")
    langs = identify_lang(first)
    first["word_count"] = first.text.apply(lambda x: len(str(x).split()))
    counts = first.word_count
    #dates = first.date
    dates = pd.to_datetime(first.date, unit='s') #Maybe erase this
    #There is a total of 17 csvs for BeerAdvocate textual ratings and 15 for RateBeer.
    if dataset == "BeerAdvocate" :
        csv_count = 17
    else :
        csv_count = 15
    for index in range(1,csv_count):
        temp = pd.read_csv(f"data/{dataset}_ratings_part_{index}.csv",low_memory=False)
        temp = temp[temp['text'] != ' nan']
        temp["word_count"] = first.text.apply(lambda x: len(str(x).split()))
        count = temp.word_count
        #date = temp.date
        date = pd.to_datetime(temp.date, unit='s')  #Maybe erase this
        temp = identify_lang(temp)
        langs = pd.concat([langs, temp])
        counts = pd.concat([counts, count])
        dates = pd.concat([dates, date])
    print("Done")
    return langs, counts, dates
### Language Identification

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "/tmp/lid.176.bin"
        self.model = fasttext.load_model("functions/lid.176.bin")

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) # returns top 2 matching languages
        return predictions

LANGUAGE = LanguageIdentification()

def identify_lang(df) :
    #TODO clean better the csvs so ' nan's are NaNs or already filtered
    df = df[df['text'] != ' nan']
    df.loc[:,"text"] = df.text.apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    series = df["text"].map(lambda x : iso_639_dict[re.search(r'[^\_]+$',LANGUAGE.predict_lang(x)[0][0]).group()])
    return series

### TODO : put this things in a separate file.
iso_639_choices = [('ab', 'Abkhaz'),
('aa', 'Afar'),
('af', 'Afrikaans'),
('ak', 'Akan'),
('ast', 'Asturian'),
('als', 'Tosk Albanian'),
('sq', 'Albanian'),
('am', 'Amharic'),
('ar', 'Arabic'),
('an', 'Aragonese'),
('hy', 'Armenian'),
('as', 'Assamese'),
('av', 'Avaric'),
('ae', 'Avestan'),
('ay', 'Aymara'),
('az', 'Azerbaijani'),
('bm', 'Bambara'),
('ba', 'Bashkir'),
('eu', 'Basque'),
('be', 'Belarusian'),
('bn', 'Bengali'),
('bh', 'Bihari'),
('bi', 'Bislama'),
('bs', 'Bosnian'),
('br', 'Breton'),
('bg', 'Bulgarian'),
('my', 'Burmese'),
('ca', 'Catalan; Valencian'),
('ch', 'Chamorro'),
('ce', 'Chechen'),
('ny', 'Chichewa; Chewa; Nyanja'),
('zh', 'Chinese'),
('cv', 'Chuvash'),
('kw', 'Cornish'),
('ceb', 'Cebuano'),
('co', 'Corsican'),
('cr', 'Cree'),
('hr', 'Croatian'),
('cs', 'Czech'),
('da', 'Danish'),
('dv', 'Divehi; Maldivian;'),
('nl', 'Dutch'),
('diq', 'Dimli'),
('dz', 'Dzongkha'),
('en', 'English'),
('eo', 'Esperanto'),
('et', 'Estonian'),
('ee', 'Ewe'),
('fo', 'Faroese'),
('fj', 'Fijian'),
('fi', 'Finnish'),
('fr', 'French'),
('ff', 'Fula'),
('gl', 'Galician'),
('ka', 'Georgian'),
('de', 'German'),
('el', 'Greek, Modern'),
('gn', 'Guaraní'),
('gu', 'Gujarati'),
('ht', 'Haitian'),
('ha', 'Hausa'),
('he', 'Hebrew (modern)'),
('hz', 'Herero'),
('hi', 'Hindi'),
('ho', 'Hiri Motu'),
('hu', 'Hungarian'),
('ia', 'Interlingua'),
('id', 'Indonesian'),
('ie', 'Interlingue'),
('ga', 'Irish'),
('ig', 'Igbo'),
('ik', 'Inupiaq'),
('io', 'Ido'),
('is', 'Icelandic'),
('it', 'Italian'),
('iu', 'Inuktitut'),
('ja', 'Japanese'),
('jv', 'Javanese'),
('kl', 'Kalaallisut'),
('kn', 'Kannada'),
('kr', 'Kanuri'),
('ks', 'Kashmiri'),
('kk', 'Kazakh'),
('km', 'Khmer'),
('ki', 'Kikuyu, Gikuyu'),
('rw', 'Kinyarwanda'),
('ky', 'Kirghiz, Kyrgyz'),
('kv', 'Komi'),
('kg', 'Kongo'),
('ko', 'Korean'),
('ku', 'Kurdish'),
('kj', 'Kwanyama, Kuanyama'),
('la', 'Latin'),
('lb', 'Luxembourgish'),
('lg', 'Luganda'),
('li', 'Limburgish'),
('ln', 'Lingala'),
('lo', 'Lao'),
('lt', 'Lithuanian'),
('lu', 'Luba-Katanga'),
('lv', 'Latvian'),
('gv', 'Manx'),
('mk', 'Macedonian'),
('mg', 'Malagasy'),
('ms', 'Malay'),
('ml', 'Malayalam'),
('mt', 'Maltese'),
('mi', 'Māori'),
('min', 'Minangkabau'),
('mr', 'Marathi (Marāṭhī)'),
('mh', 'Marshallese'),
('mn', 'Mongolian'),
('na', 'Nauru'),
('nv', 'Navajo, Navaho'),
('nb', 'Norwegian Bokmål'),
('nd', 'North Ndebele'),
('nds', 'Low German'),
('ne', 'Nepali'),
('new', 'Newari'),
('ng', 'Ndonga'),
('nn', 'Norwegian Nynorsk'),
('no', 'Norwegian'),
('ii', 'Nuosu'),
('nr', 'South Ndebele'),
('oc', 'Occitan'),
('oj', 'Ojibwe, Ojibwa'),
('cu', 'Old Church Slavonic'),
('om', 'Oromo'),
('or', 'Oriya'),
('os', 'Ossetian, Ossetic'),
('pa', 'Panjabi, Punjabi'),
('pi', 'Pāli'),
('fa', 'Persian'),
('pl', 'Polish'),
('ps', 'Pashto, Pushto'),
('pt', 'Portuguese'),
('qu', 'Quechua'),
('rm', 'Romansh'),
('rn', 'Kirundi'),
('ro', 'Romanian, Moldavan'),
('ru', 'Russian'),
('sa', 'Sanskrit (Saṁskṛta)'),
('sc', 'Sardinian'),
('sd', 'Sindhi'),
('se', 'Northern Sami'),
('sh', 'Serbo-Croatian'),                   
('sm', 'Samoan'),
('sg', 'Sango'),
('sr', 'Serbian'),
('gd', 'Scottish Gaelic'),
('sn', 'Shona'),
('si', 'Sinhala, Sinhalese'),
('sk', 'Slovak'),
('sl', 'Slovene'),
('so', 'Somali'),
('st', 'Southern Sotho'),
('es', 'Spanish; Castilian'),
('su', 'Sundanese'),
('sw', 'Swahili'),
('ss', 'Swati'),
('sv', 'Swedish'),
('ta', 'Tamil'),
('te', 'Telugu'),
('tg', 'Tajik'),
('th', 'Thai'),
('ti', 'Tigrinya'),
('bo', 'Tibetan'),
('tk', 'Turkmen'),
('tl', 'Tagalog'),
('tn', 'Tswana'),
('to', 'Tonga'),
('tr', 'Turkish'),
('ts', 'Tsonga'),
('tt', 'Tatar'),
('tw', 'Twi'),
('ty', 'Tahitian'),
('ug', 'Uighur, Uyghur'),
('uk', 'Ukrainian'),
('ur', 'Urdu'),
('uz', 'Uzbek'),
('ve', 'Venda'),
('vi', 'Vietnamese'),
('vo', 'Volapük'),
('wa', 'Walloon'),
('cy', 'Welsh'),
('wo', 'Wolof'),
('fy', 'Western Frisian'),
('xh', 'Xhosa'),
('yi', 'Yiddish'),
('yo', 'Yoruba'),
('za', 'Zhuang, Chuang'),
('zu', 'Zulu'),
('war', 'Waray'),
('wuu', 'Wu Chinese'),
('xmf', 'Mingrelian'),
('yue', 'Yue Chinese'),
]


iso_639_dict = {}

# iterating over the tuples lists
for (key, value) in iso_639_choices:
   # setting the default value as list([])
   # appending the current value
   iso_639_dict[key] = (value)
