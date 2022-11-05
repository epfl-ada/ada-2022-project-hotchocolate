#!/usr/bin/python3

import pandas as pd
import tarfile
import gzip
import re
MAX_CSV_SIZE = 1000000
CHUNK_SIZE = 200
def fetch_satellite_data():
    return 0

def fetch_csv(dataset_path, csv=None):
    try:
        with tarfile.open(dataset_path) as tar:
            csv_list = []
            for filename in tar.getnames():
                if ".csv" in filename:
                    csv_list.append(filename)
            dataframe_list = []
            for csv in csv_list:    
                with tar.extractfile(csv) as file:
                    dataframe_list.append(pd.read_csv(file))
        return dataframe_list
    except: 
        print("Euh, no file was found in this path")

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
                            df.to_csv(f"../data/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
                            del review_dict #Just to not kill my memory please disregard :)
                            review_dict = {}
                            print(f"Dumping data to csv number {csv_count}...")
                            csv_count += 1
                            if early_stop and csv_count == early_stop : 
                                break
                    else:
                        (_,value) = line.split(":", 1)
                        review.append(value.rstrip())
    
    df = pd.DataFrame.from_dict(review_dict, orient="index")
    df.to_csv(f"../data/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
    del review_dict #Just to not kill my memory please disregard :)
    print(f"Dumping data to csv number {csv_count}...")
    print(i)
    print("Success!")
    if early_stop :
        return 1
    else : 
        return df
#    except: 
#        print("Euh, no file was found in this path")
