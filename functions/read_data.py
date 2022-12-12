#!/usr/bin/python3

import pandas as pd
import tarfile
import gzip
import re
import requests
import urllib.request
from bs4 import BeautifulSoup
import json
MAX_CSV_SIZE = 1000000
CHUNK_SIZE = 200
def fetch_satellite_data():
    return 0
COLUMNS_NAMES = ["beer_name","beer_id","brewery_name","brewery_id","style","abv","date","username","user_id","appearance","aroma","palate","taste","overall","rating","text"]

def fetch_csv(dataset_path, name):
    with tarfile.open(dataset_path) as tar:
        dataframe = pd.DataFrame()
        for filename in tar.getnames():
            if name in filename:   
                with tar.extractfile(filename) as file:
                    dataframe = pd.read_csv(file)
    return dataframe


def fetch_reviews(dataset_path, max_csv_size = MAX_CSV_SIZE,early_stop = 0):
    tarfile_name = re.search("[ \w-]+?(?=\.)",dataset_path)[0]
    with tarfile.open(dataset_path) as tar:
        datadumps = [filename for filename in tar.getnames() if "txt.gz" in filename]
        print(datadumps)
        filename = input("Please choose a file from the list above to open: ")
        with tar.extractfile(filename) as file:
            with gzip.open(file,'rt') as f:
                review = []
                review_dict = {}
                row_count = 0
                csv_count = 0
                for line in f:
                    if len(line) < 2:
                        review_dict[row_count] = review
                        row_count += 1
                        review = []
                        if row_count % max_csv_size == 0:
                            df = pd.DataFrame.from_dict(review_dict, orient="index")
                            df.columns = COLUMNS_NAMES
                            df.to_csv(f"/DATA/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
                            del review_dict #Just to not kill my memory please disregard :)
                            review_dict = {}
                            print(f"Dumping data to csv number {csv_count}...")
                            csv_count += 1
                            if early_stop and csv_count == early_stop : 
                                break
                    else:
                        (key,value) = line.split(":", 1)
                        #BeerAdvocate has one column called "review" that is useless and makes everything harder
                        if key != "review":
                            review.append(value.rstrip())
    
    df = pd.DataFrame.from_dict(review_dict, orient="index")
    df.to_csv(f"/DATA/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
    del review_dict #Just to not kill my memory please disregard :)
    print(f"Dumping data to csv number {csv_count}...")
    print("Success!")
    if early_stop :
        return 1
    else : 
        return df
#    except: 
#        print("Euh, no file was found in this path")

def fetch_satellite_df():

    url = "https://satellite.bar/bar/"
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    beer = re.findall('var djangoBoissons = (.*?);\s*$', soup.prettify(), re.M)
    beerJson = beer[0]
    satellite_dict = json.loads(beerJson)
    satellite_df = pd.DataFrame(satellite_dict)

    return satellite_df